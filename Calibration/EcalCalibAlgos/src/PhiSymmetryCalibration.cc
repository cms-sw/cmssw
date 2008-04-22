#include "Calibration/EcalCalibAlgos/interface/PhiSymmetryCalibration.h"

// System include files
#include <memory>

// Framework
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

// Conditions database
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"
#include "Calibration/Tools/interface/calibXMLwriter.h"

// Geometry
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

using namespace std;
#include <fstream>
#include <iostream>
#include "TH2F.h"

//_____________________________________________________________________________

PhiSymmetryCalibration::PhiSymmetryCalibration(const edm::ParameterSet& iConfig) :
  theMaxLoops( iConfig.getUntrackedParameter<unsigned int>("maxLoops",0) ),
  ecalHitsProducer_( iConfig.getParameter< std::string > ("ecalRecHitsProducer") ),
  barrelHits_( iConfig.getParameter< std::string > ("barrelHitCollection") ),
  endcapHits_( iConfig.getParameter< std::string > ("endcapHitCollection") ),
  eCut_barl_( iConfig.getParameter< double > ("eCut_barrel") ),
  eCut_endc_( iConfig.getParameter< double > ("eCut_endcap") ),
  eventSet_( iConfig.getParameter< int > ("eventSet") )
{

  edm::LogWarning("Calibration") << "[PhiSymmetryCalibration] Constructor called ...";

  theParameterSet=iConfig;

}


//_____________________________________________________________________________
// Close files, etc.

PhiSymmetryCalibration::~PhiSymmetryCalibration()
{

}

//_____________________________________________________________________________
// Initialize algorithm

void PhiSymmetryCalibration::beginJob( const edm::EventSetup& iSetup )
{

  edm::LogWarning("Calibration") << "[PhiSymmetryCalibration] At begin job ...";

  // initialize arrays

  for (int sign=0; sign<2; sign++) {
    for (int ieta=0; ieta<85; ieta++) {
      for (int iphi=0; iphi<360; iphi++) {
	etsum_barl_[ieta][iphi][sign]=0.;
      }
    }
    for (int ix=0; ix<100; ix++) {
      for (int iy=0; iy<100; iy++) {
	etsum_endc_[ix][iy][sign]=0.;
      }
    }
  }


  for (int ieta=0; ieta<85; ieta++) cellEta_[ieta]=0.;

  for (int ix=0; ix<100; ix++) {
    for (int iy=0; iy<100; iy++) {
      cellPos_[ix][iy] = GlobalPoint(0.,0.,0.);
      cellPhi_[ix][iy]=0.;
      cellArea_[ix][iy]=0.;
      endcapRing_[ix][iy]=-1;
    }
  }

  for (int imiscal=0; imiscal<21; imiscal++) {
    miscal_[imiscal]=.95+float(imiscal)/200.;
    for (int ieta=0; ieta<85; ieta++) etsum_barl_miscal_[imiscal][ieta]=0.;
    for (int ring=0; ring<39; ring++) etsum_endc_miscal_[imiscal][ring]=0.;
  }

  // get initial constants out of DB
  EcalIntercalibConstantMap imap;
  if (eventSet_==0) {
    edm::ESHandle<EcalIntercalibConstants> pIcal;
    try {
      iSetup.get<EcalIntercalibConstantsRcd>().get(pIcal);
      std::cout << "Taken EcalIntercalibConstants" << std::endl;
      imap = pIcal.product()->getMap();
      std::cout << "imap.size() = " << imap.size() << std::endl;
    } catch ( std::exception& ex ) {     
      std::cerr << "Error! can't get EcalIntercalibConstants " << std::endl;
    }
  }

  // get the ecal geometry:
  edm::ESHandle<CaloGeometry> geoHandle;
  iSetup.get<IdealGeometryRecord>().get(geoHandle);
  const CaloGeometry& geometry = *geoHandle;
  const CaloSubdetectorGeometry *barrelGeometry = geometry.getSubdetectorGeometry(DetId::Ecal, EcalBarrel);
  const CaloSubdetectorGeometry *endcapGeometry = geometry.getSubdetectorGeometry(DetId::Ecal, EcalEndcap);;


  // loop over all barrel crystals
  barrelCells = geometry.getValidDetIds(DetId::Ecal, EcalBarrel);
  std::vector<DetId>::const_iterator barrelIt;
  for (barrelIt=barrelCells.begin(); barrelIt!=barrelCells.end(); barrelIt++) {
    EBDetId eb(*barrelIt);

    if (eventSet_==0) {
      // get the initial calibration constants
      EcalIntercalibConstantMap::const_iterator itcalib = imap.find(eb.rawId());
      if ( itcalib == imap.end() ) {
              // FIXME -- throw error
      }
      EcalIntercalibConstant calib = (*itcalib);
      int sign = eb.zside()>0 ? 1 : 0;
      oldCalibs_barl[abs(eb.ieta())-1][eb.iphi()-1][sign] = calib;
      if (eb.iphi()==1) std::cout << "Read old constant for crystal "
				  << " (" << eb.ieta() << "," << eb.iphi()
				  << ") : " << calib << std::endl;
    }

    // store eta value for each ring
    if (eb.ieta()>0 &&eb.iphi()==1) {
      const CaloCellGeometry *cellGeometry = barrelGeometry->getGeometry(*barrelIt);
      cellEta_[eb.ieta()-1] = cellGeometry->getPosition().eta();
    }
  }

  // loop over all endcap crystals
  endcapCells = geometry.getValidDetIds(DetId::Ecal, EcalEndcap);
  std::vector<DetId>::const_iterator endcapIt;
  for (endcapIt=endcapCells.begin(); endcapIt!=endcapCells.end(); endcapIt++) {
    const CaloCellGeometry *cellGeometry = endcapGeometry->getGeometry(*endcapIt);
    EEDetId ee(*endcapIt);
    int ix=ee.ix()-1;
    int iy=ee.iy()-1;

    if (eventSet_==0) {
      // get the initial calibration constants
      EcalIntercalibConstantMap::const_iterator itcalib = imap.find(ee.rawId());
      if ( itcalib == imap.end() ) {
              // FIXME -- throw error
      }
      EcalIntercalibConstant calib = (*itcalib);
      int sign = ee.zside()>0 ? 1 : 0;
      oldCalibs_endc[ix][iy][sign] = calib;
      if (ix==49) std::cout << "Read old constant for crystal "
			    << " (" << ix << "," << iy
			    << ") : " << calib << std::endl;
    }

    // store all crystal positions
    cellPos_[ix][iy] = cellGeometry->getPosition();
    cellPhi_[ix][iy] = cellGeometry->getPosition().phi();



    // calculate and store eta-phi area for each crystal front face
    const CaloCellGeometry::CornersVec& cellCorners (cellGeometry->getCorners()) ;
    cellArea_[ix][iy]=0.;
    for (int i=0; i<4; i++) {
      int iplus1 = i==3 ? 0 : i+1;
      cellArea_[ix][iy] += 
	cellCorners[i].eta()*float(cellCorners[iplus1].phi()) - 
	cellCorners[iplus1].eta()*float(cellCorners[i].phi());
    }
    cellArea_[ix][iy] = cellArea_[ix][iy]/2.;

  }    


  // get eta for each endcap ring
  float eta_ring[39];
  for (int ring=0; ring<39; ring++) {
    eta_ring[ring]=cellPos_[ring][50].eta();
  }

  // get eta boundaries for each endcap ring
  etaBoundary_[0]=1.479;
  etaBoundary_[39]=4.;
  for (int ring=1; ring<39; ring++) {
    etaBoundary_[ring]=(eta_ring[ring]+eta_ring[ring-1])/2.;
  }

  // determine to which ring each endcap crystal belongs,
  // the number of crystals in each ring,
  // and the mean eta-phi area of the crystals in each ring
  for (int ring=0; ring<39; ring++) {
    nRing_[ring]=0;
    meanCellArea_[ring]=0.;
    for (int ix=0; ix<100; ix++) {
      for (int iy=0; iy<100; iy++) {
	if (fabs(cellPos_[ix][iy].eta())>etaBoundary_[ring] &&
	    fabs(cellPos_[ix][iy].eta())<etaBoundary_[ring+1]) {
	  meanCellArea_[ring]+=cellArea_[ix][iy];
	  endcapRing_[ix][iy]=ring;
	  nRing_[ring]++;
	}
      }
    }

    meanCellArea_[ring]/=nRing_[ring];

    std::cout << nRing_[ring] << " crystals with mean area " << meanCellArea_[ring] << " in endcap ring " << ring << " (" << etaBoundary_[ring] << "<eta<" << etaBoundary_[ring+1] << ")" << std::endl;
  }


    // book and fill and output histograms of ET sum vs phi

  
    for (int ring=0; ring<39; ring++) {
      char ch[3];
      sprintf(ch, "etsumVsPhi_neg1_%i", ring+1);
      
      etsumVsPhi_histos1[ring][0] = 
			       new TH1F(ch,"",nRing_[ring],.5,nRing_[ring]+.5);


      sprintf(ch, "etsumVsPhi_pos1_%i", ring+1);
   
      etsumVsPhi_histos1[ring][1]=new TH1F(ch,"",nRing_[ring],.5,nRing_[ring]+.5);
    }


   for (int ring=0; ring<39; ring++) {
      char ch[3];
      sprintf(ch, "etsumVsPhi_neg_%i", ring+1);
      etsumVsPhi_histos[ring][0] = new TH1F(ch,"",nRing_[ring],.5,nRing_[ring]+.5);
      sprintf(ch, "etsumVsPhi_pos_%i", ring+1);
      etsumVsPhi_histos[ring][1] = new TH1F(ch,"",nRing_[ring],.5,nRing_[ring]+.5);
      sprintf(ch, "areaVsPhi_%i", ring+1);
      areaVsPhi_histos[ring] = new TH1F(ch,"",nRing_[ring],.5,nRing_[ring]+.5);
      sprintf(ch, "etaVsPhi_%i", ring+1);
      etaVsPhi_histos[ring] = new TH1F(ch,"",nRing_[ring],.5,nRing_[ring]+.5);
    }


for (int ix=0; ix<100; ix++) {
      for (int iy=0; iy<100; iy++) {
	int ring = endcapRing_[ix][iy];
	if (ring!=-1) {
	  int iphi_endc=0;
	  for (int ip=0; ip<nRing_[ring]; ip++) {
	    if (cellPhi_[ix][iy]==phi_endc[ip][ring]) iphi_endc=ip;
	  }
	  areaVsPhi_histos[ring]->Fill(iphi_endc,cellArea_[ix][iy]);
	  etaVsPhi_histos[ring]->Fill(iphi_endc,cellPos_[ix][iy].eta());
	  for (int sign=0; sign<2; sign++) {
	    etsumVsPhi_histos[ring][sign]->Fill(iphi_endc,etsum_endc_[ix][iy][sign]);
	  }
	}
      }
    }



 //
    for (int ring=0; ring<39; ring++) {
      for (int i=0; i<300; i++) phi_endc[i][ring]=0.;
      float philast=-999.;
      for (int ip=0; ip<nRing_[ring]; ip++) {
	float phimin=999.;
	for (int ix=0; ix<100; ix++) {
	  for (int iy=0; iy<100; iy++) {
	    if (endcapRing_[ix][iy]==ring) {
	      if (cellPhi_[ix][iy]<phimin && cellPhi_[ix][iy]>philast) {
		phimin=cellPhi_[ix][iy];
	      }
	    }
	  }
	}
	phi_endc[ip][ring]=phimin;
	philast=phimin;
      }
    }

}

//_____________________________________________________________________________
// Terminate algorithm

void PhiSymmetryCalibration::endJob()
{

  edm::LogWarning("Calibration") << "[PhiSymmetryCalibration] At end of job";

  if (eventSet_==1) {
    // calculate factors to convert from fractional deviation of ET sum from 
    // the mean to the estimate of the miscalibration factor
    getKfactors();


    std::ofstream k_barl_out("k_barl.dat", ios::out);
    for (int ieta=0; ieta<85; ieta++)
      k_barl_out << ieta << " " << k_barl_[ieta] << endl;
    k_barl_out.close();

    std::ofstream k_endc_out("k_endc.dat", ios::out);
    for (int ring=0; ring<39; ring++)
      k_endc_out << ring << " " << k_endc_[ring] << endl;
    k_endc_out.close();
  }

  if (eventSet_!=0) {

    //output ET sums

    std::ofstream etsum_barl_out("etsum_barl.dat",ios::app);
    for (int ieta=0; ieta<85; ieta++) {
      for (int iphi=0; iphi<360; iphi++) {
	for (int sign=0; sign<2; sign++) {
	  etsum_barl_out << eventSet_ << " " << ieta << " " << iphi << " " << sign << "  "<< etsum_barl_[ieta][iphi][sign] << endl;
	}
      }
    }
    etsum_barl_out.close();

    std::ofstream etsum_endc_out("etsum_endc.dat",ios::app);
    for (int ix=0; ix<100; ix++) {
      for (int iy=0; iy<100; iy++) {
	int ring = endcapRing_[ix][iy];
	if (ring!=-1) {
	  for (int sign=0; sign<2; sign++) {
	    etsum_endc_out << eventSet_ << " " << ix << " " << iy << " " << sign << " " << etsum_endc_[ix][iy][sign] << endl;
	  }
	}
      }
    }
    etsum_endc_out.close();

  } else {

    for (int ieta=0; ieta<85; ieta++) {
      for (int iphi=0; iphi<360; iphi++) {
	for (int sign=0; sign<2; sign++) {
	  etsum_barl_[ieta][iphi][sign]=0.;
	}
      }
    }

    for (int ix=0; ix<100; ix++) {
      for (int iy=0; iy<100; iy++) {
	for (int sign=0; sign<2; sign++) {
	  etsum_endc_[ix][iy][sign]=0.;
	}
      }
    }

    //read in ET sums

    int ieta,iphi,sign,ix,iy,dummy;
    double etsum;
    std::ifstream etsum_barl_in("etsum_barl.dat", ios::in);
    while (!etsum_barl_in.eof()) {
      etsum_barl_in >> dummy;
      etsum_barl_in >> ieta;
      etsum_barl_in >> iphi;
      etsum_barl_in >> sign;
      etsum_barl_in >> etsum;
      etsum_barl_[ieta][iphi][sign]+=etsum;
    }
    etsum_barl_in.close();

    std::ifstream etsum_endc_in("etsum_endc.dat", ios::in);
    while (!etsum_endc_in.eof()) {
      etsum_endc_in >> dummy;
      etsum_endc_in >> ix;
      etsum_endc_in >> iy;
      etsum_endc_in >> sign;
      etsum_endc_in >> etsum;
      etsum_endc_[ix][iy][sign]+=etsum;
    }
    etsum_endc_in.close();

    std::ifstream k_barl_in("k_barl.dat", ios::in);
    for (int ieta=0; ieta<85; ieta++) {
      k_barl_in >> dummy;
      k_barl_in >> k_barl_[ieta];
    }
    k_barl_in.close();

    std::ifstream k_endc_in("k_endc.dat", ios::in);
    for (int ring=0; ring<39; ring++) {
      k_endc_in >> dummy;
      k_endc_in >> k_endc_[ring];
    }
    k_endc_in.close();



   

    for (int ix=0; ix<100; ix++) {
      for (int iy=0; iy<100; iy++) {

	int ring = endcapRing_[ix][iy];
	if (ring!=-1) {
	  int iphi_endc=0;
	  for (int ip=0; ip<nRing_[ring]; ip++) {
	    if (cellPhi_[ix][iy]==phi_endc[ip][ring]) iphi_endc=ip;
	  }
	  for (int sign=0; sign<2; sign++) {

	    etsumVsPhi_histos1[ring][sign]->Fill(iphi_endc,etsum_endc_[ix][iy][sign]);
	 
	  }
	}
      }
    }

    for (int ix=0; ix<100; ix++) {
      for (int iy=0; iy<100; iy++) {

	int ring = endcapRing_[ix][iy];
	if (ring!=-1) {
	  for (int sign=0; sign<2; sign++) {

	    etsum_endc_[ix][iy][sign]*=meanCellArea_[ring]/cellArea_[ix][iy];
	  }
	}
      }
    }
 

    TFile f1("etsumVsPhi.root","recreate");
    for (int ring=0; ring<39; ring++) {
      areaVsPhi_histos[ring]->Write();
      etaVsPhi_histos[ring]->Write();
      for (int sign=0; sign<2; sign++) {
	etsumVsPhi_histos1[ring][sign]->Write();
	etsumVsPhi_histos[ring][sign]->Write();
      }
    }
    f1.Close();

    // fill output ET sum histograms
    fillHistos();

    std::ofstream etsumMean_barl_out("etsumMean_barl.dat",ios::out);
    for (int ieta=0; ieta<85; ieta++) {
      etsumMean_barl_out << cellEta_[ieta] << " " << etsumMean_barl_[ieta] << endl;
    }
    etsumMean_barl_out.close();

    std::ofstream etsumMean_endc_out("etsumMean_endc.dat",ios::out);
    for (int ring=0; ring<39; ring++) {
      etsumMean_endc_out << cellPos_[ring][50].eta() << " " << etsumMean_endc_[ring] << endl;
    }
    etsumMean_endc_out.close();

    std::ofstream area_out("area.dat",ios::out);
    for (int ring=0; ring<39; ring++) {
      area_out << meanCellArea_[ring] << endl;
    }
    area_out.close();

    // Determine barrel calibration constants
    float epsilon_M_barl[85][360][2];
    for (int ieta=0; ieta<85; ieta++) {
      for (int iphi=0; iphi<360; iphi++) {
	for (int sign=0; sign<2; sign++) {
	  float etsum = etsum_barl_[ieta][iphi][sign];
	  float epsilon_T = (etsum/etsumMean_barl_[ieta])-1;
	  epsilon_M_barl[ieta][iphi][sign] = epsilon_T/k_barl_[ieta];
	}
      }
    }

    // Determine endcap calibration constants
    float epsilon_M_endc[100][100][2];
    for (int ix=0; ix<100; ix++) {
      for (int iy=0; iy<100; iy++) {
	int ring = endcapRing_[ix][iy];
	if (ring!=-1) {
	  for (int sign=0; sign<2; sign++) {
	    float etsum = etsum_endc_[ix][iy][sign];
	    float epsilon_T = (etsum/etsumMean_endc_[ring])-1;
	  epsilon_M_endc[ix][iy][sign] = epsilon_T/k_endc_[ring];
	  }
	} else {
	  epsilon_M_endc[ix][iy][0] = 0.;
	  epsilon_M_endc[ix][iy][1] = 0.;
	}
      }
    }

    // Write new calibration constants

    calibXMLwriter barrelWriter(EcalBarrel);
    calibXMLwriter endcapWriter(EcalEndcap);

    double newCalibs_barl[100][100][2];
    double newCalibs_endc[100][100][2];

    TH1F* miscal_resid_barl_histos[85];
    TH2F* correl_barl_histos[85];
    for (int ieta=0; ieta<85; ieta++) {
      char ch[3];
      sprintf(ch, "mr_barl_%i", ieta+1);
      miscal_resid_barl_histos[ieta] = new TH1F(ch,"",50,.8,1.2);
      sprintf(ch, "co_barl_%i", ieta+1);
      correl_barl_histos[ieta] = new TH2F(ch,"",50,.8,1.2,50,.8,1.2);
    }

    TH1F* miscal_resid_endc_histos[39];
    TH2F* correl_endc_histos[39];
    for (int ring=0; ring<39; ring++) {
      char ch[3];
      sprintf(ch, "mr_endc_%i", ring+1);
      miscal_resid_endc_histos[ring] = new TH1F(ch,"",50,.8,1.2);
      sprintf(ch, "co_endc_%i", ring+1);
      correl_endc_histos[ring] = new TH2F(ch,"",50,.8,1.2,50,.8,1.2);
    }

    std::vector<DetId>::const_iterator barrelIt=barrelCells.begin();
    for (; barrelIt!=barrelCells.end(); barrelIt++) {
      EBDetId eb(*barrelIt);
      int ieta = abs(eb.ieta())-1;
      int iphi = eb.iphi()-1;
      int sign = eb.zside()>0 ? 1 : 0;
      newCalibs_barl[ieta][iphi][sign] = 
                                    oldCalibs_barl[ieta][iphi][sign]/
                                 (1+epsilon_M_barl[ieta][iphi][sign]);
      barrelWriter.writeLine(eb,newCalibs_barl[ieta][iphi][sign]);
      miscal_resid_barl_histos[ieta]->Fill(newCalibs_barl[ieta][iphi][sign]);
      correl_barl_histos[ieta]->Fill(oldCalibs_barl[ieta][iphi][sign],1+epsilon_M_barl[ieta][iphi][sign]);
      if (iphi==1) {
	std::cout << "Calib constant for barrel crystal "
		  << " (" << eb.ieta() << "," << eb.iphi() << ") changed from "
		  << oldCalibs_barl[ieta][iphi][sign] << " to "
		  << newCalibs_barl[ieta][iphi][sign] << std::endl;
      }
    }

    std::vector<DetId>::const_iterator endcapIt=endcapCells.begin();
    for (; endcapIt!=endcapCells.end(); endcapIt++) {
      EEDetId ee(*endcapIt);
      int ix = ee.ix()-1;
      int iy = ee.iy()-1;
      int sign = ee.zside()>0 ? 1 : 0;
      newCalibs_endc[ix][iy][sign] = 
                                    oldCalibs_endc[ix][iy][sign]/
                                 (1+epsilon_M_endc[ix][iy][sign]);
      endcapWriter.writeLine(ee,newCalibs_endc[ix][iy][sign]);
      miscal_resid_endc_histos[endcapRing_[ix][iy]]->Fill(newCalibs_endc[ix][iy][sign]);
      correl_endc_histos[endcapRing_[ix][iy]]->Fill(oldCalibs_endc[ix][iy][sign],1+epsilon_M_endc[ix][iy][sign]);
      if (ix==50) {
	std::cout << "Calib constant for endcap crystal "
		  << " (" << ix << "," << iy << "," << sign << ") changed from "
		  << oldCalibs_endc[ix][iy][sign] << " to "
		  << newCalibs_endc[ix][iy][sign] << std::endl;
      }
    }

    //double prec_barl[85],prec_endc[39];

    // Output histograms of residual miscalibrations
    TFile f("PhiSymmetryCalibration_miscal_resid.root","recreate");
    for (int ieta=0; ieta<85; ieta++) {
      miscal_resid_barl_histos[ieta]->Fit("gaus");
      miscal_resid_barl_histos[ieta]->Write();
      //prec_barl[ieta] = miscal_resid_barl_histos[ieta]->GetFunction("gaus")->GetParameter(2);
      correl_barl_histos[ieta]->Write();
    }
    for (int ring=0; ring<39; ring++) {
      miscal_resid_endc_histos[ring]->Fit("gaus");
      miscal_resid_endc_histos[ring]->Write();
      //prec_endc[ring] = miscal_resid_endc_histos[ring]->GetFunction("gaus")->GetParameter(2);
      correl_endc_histos[ring]->Write();
    }
    f.Close();

    /*
    std::ofstream prec_barl_out("prec_barl.dat", ios::out);
    for (int ieta=0; ieta<85; ieta++)
      prec_barl_out << cellEta_[ieta] << " " << prec_barl[ieta] << endl;
    prec_barl_out.close();

    std::ofstream prec_endc_out("prec_endc.dat", ios::out);
    for (int ring=0; ring<39; ring++)
      prec_endc_out << cellPos_[ring][50].eta() << " " << prec_endc[ring] << endl;
    prec_endc_out.close();
    */
  }

}

//_____________________________________________________________________________
// Called at each event

void PhiSymmetryCalibration::analyze( const edm::Event& event, 
  const edm::EventSetup& setup )
{
  using namespace edm;
  using namespace std;

  nevent++;

  edm::LogInfo("Calibration") << "[PhiSymmetryCalibration] New Event --------------------------------------------------------------";

  if ((nevent<100 && nevent%10==0) 
      ||(nevent<1000 && nevent%100==0) 
      ||(nevent<10000 && nevent%100==0) 
      ||(nevent<100000 && nevent%1000==0) 
      ||(nevent<10000000 && nevent%1000==0))
    edm::LogWarning("Calibration") << "[PhiSymmetryCalibration] Events processed: "<<nevent;

  Handle<EBRecHitCollection> barrelRecHitsHandle;
  Handle<EERecHitCollection> endcapRecHitsHandle;

  event.getByLabel(ecalHitsProducer_,barrelHits_,barrelRecHitsHandle);
  if (!barrelRecHitsHandle.isValid()) {
    LogDebug("") << "PhiSymmetryCalibration: Error! can't get product!" << std::endl;
  }

  event.getByLabel(ecalHitsProducer_,endcapHits_,endcapRecHitsHandle);
  if (!endcapRecHitsHandle.isValid()) {
    LogDebug("") << "PhiSymmetryCalibration: Error! can't get product!" << std::endl;
  }

  //Select interesting EcalRecHits (barrel)
  EBRecHitCollection::const_iterator itb;
  for (itb=barrelRecHitsHandle->begin(); itb!=barrelRecHitsHandle->end(); itb++) {
    EBDetId hit = EBDetId(itb->id());
    float eta = cellEta_[abs(hit.ieta())-1];
    float et = itb->energy()/cosh(eta);
    float et_thr = eCut_barl_/cosh(eta);
    et_thr*=1.05;
    if (et > et_thr && et < et_thr+0.8) {
      int sign = hit.ieta()>0 ? 1 : 0;
      etsum_barl_[abs(hit.ieta())-1][hit.iphi()-1][sign] += et;
    }

    if (eventSet_==1) {
      //Apply a miscalibration to all crystals and increment the 
      //ET sum, combined for all crystals
      for (int imiscal=0; imiscal<21; imiscal++) {
	if (miscal_[imiscal]*et > et_thr && miscal_[imiscal]*et < et_thr+0.8) {
	  etsum_barl_miscal_[imiscal][abs(hit.ieta())-1] += miscal_[imiscal]*et;
	}
      }
    }

  }

  //Select interesting EcalRecHits (endcaps)
  EERecHitCollection::const_iterator ite;
  for (ite=endcapRecHitsHandle->begin(); ite!=endcapRecHitsHandle->end(); ite++) {
    EEDetId hit = EEDetId(ite->id());
    float eta = cellPos_[hit.ix()-1][hit.iy()-1].eta();
    float et = ite->energy()/cosh(eta);
    float et_thr = eCut_endc_/cosh(eta);
    et_thr*=1.05;
    if (et > et_thr && et < et_thr+0.8) {
      int sign = hit.zside()>0 ? 1 : 0;
      etsum_endc_[hit.ix()-1][hit.iy()-1][sign] += et;
    }

    if (eventSet_==1) {
      //Apply a miscalibration to all crystals and increment the 
      //ET sum, combined for all crystals
      for (int imiscal=0; imiscal<21; imiscal++) {
	if (miscal_[imiscal]*et > et_thr && miscal_[imiscal]*et < et_thr+0.8) {
	  int ring = endcapRing_[hit.ix()-1][hit.iy()-1];
	  etsum_endc_miscal_[imiscal][ring] += miscal_[imiscal]*et*meanCellArea_[ring]/cellArea_[hit.ix()-1][hit.iy()-1];
	}
      }
    }

  }

}

// ----------------------------------------------------------------------------

void PhiSymmetryCalibration::getKfactors()
{

  float epsilon_T[21];
  float epsilon_M[21];

  for (int ieta=0; ieta<85; ieta++) {
    for (int imiscal=0; imiscal<21; imiscal++) {
      epsilon_T[imiscal] = etsum_barl_miscal_[imiscal][ieta]/etsum_barl_miscal_[10][ieta] - 1.;
      epsilon_M[imiscal]=miscal_[imiscal]-1.;
    }
    k_barl_graph_[ieta] = new TGraph (21,epsilon_M,epsilon_T);
    k_barl_graph_[ieta]->Fit("pol1");

    char ch[3];
    sprintf(ch, "k_barl_%i", ieta+1);
    k_barl_plot_[ieta] = new TCanvas(ch,"");
    k_barl_plot_[ieta]->SetFillColor(10);
    k_barl_plot_[ieta]->SetGrid();
    k_barl_graph_[ieta]->SetMarkerSize(1.);
    k_barl_graph_[ieta]->SetMarkerColor(4);
    k_barl_graph_[ieta]->SetMarkerStyle(20);
    k_barl_graph_[ieta]->GetXaxis()->SetLimits(-.06,.06);
    k_barl_graph_[ieta]->GetXaxis()->SetTitleSize(.05);
    k_barl_graph_[ieta]->GetYaxis()->SetTitleSize(.05);
    k_barl_graph_[ieta]->GetXaxis()->SetTitle("#epsilon_{M}");
    k_barl_graph_[ieta]->GetYaxis()->SetTitle("#epsilon_{T}");
    k_barl_graph_[ieta]->Draw("AP");

    k_barl_[ieta] = k_barl_graph_[ieta]->GetFunction("pol1")->GetParameter(1);
    std::cout << "k_barl_[" << ieta << "]=" << k_barl_[ieta] << std::endl;
  }

  for (int ring=0; ring<39; ring++) {
    for (int imiscal=0; imiscal<21; imiscal++) {
      epsilon_T[imiscal] = etsum_endc_miscal_[imiscal][ring]/etsum_endc_miscal_[10][ring] - 1.;
      epsilon_M[imiscal]=miscal_[imiscal]-1.;
    }
    k_endc_graph_[ring] = new TGraph (21,epsilon_M,epsilon_T);
    k_endc_graph_[ring]->Fit("pol1");

    char ch[3];
    sprintf(ch, "k_endc_%i", ring+1);
    k_endc_plot_[ring] = new TCanvas(ch,"");
    k_endc_plot_[ring]->SetFillColor(10);
    k_endc_plot_[ring]->SetGrid();
    k_endc_graph_[ring]->SetMarkerSize(1.);
    k_endc_graph_[ring]->SetMarkerColor(4);
    k_endc_graph_[ring]->SetMarkerStyle(20);
    k_endc_graph_[ring]->GetXaxis()->SetLimits(-.06,.06);
    k_endc_graph_[ring]->GetXaxis()->SetTitleSize(.05);
    k_endc_graph_[ring]->GetYaxis()->SetTitleSize(.05);
    k_endc_graph_[ring]->GetXaxis()->SetTitle("#epsilon_{M}");
    k_endc_graph_[ring]->GetYaxis()->SetTitle("#epsilon_{T}");
    k_endc_graph_[ring]->Draw("AP");

    k_endc_[ring] = k_endc_graph_[ring]->GetFunction("pol1")->GetParameter(1);
    std::cout << "k_endc_[" << ring << "]=" << k_endc_[ring] << std::endl;
  }
 
  TFile f("PhiSymmetryCalibration_kFactors.root","recreate");
  for (int ieta=0; ieta<85; ieta++) k_barl_plot_[ieta]->Write();
  for (int ring=0; ring<39; ring++) k_endc_plot_[ring]->Write();
  f.Close();

}

void PhiSymmetryCalibration::fillHistos()
{
  TFile f("PhiSymmetryCalibration.root","recreate");

  for (int ieta=0; ieta<85; ieta++) {

    // Determine ranges of ET sums to get histo bounds and book histos (barrel)
    float low=999999.;
    float high=0.;
    for (int iphi=0; iphi<360; iphi++) {
      for (int sign=0; sign<2; sign++) {
	float etsum = etsum_barl_[ieta][iphi][sign];
	if (etsum<low) low=etsum;
	if (etsum>high) high=etsum;
      }
    }
    char ch[3];
    sprintf(ch, "etsum_barl_%i", ieta+1);
    etsum_barl_histos_[ieta]=new TH1F(ch,"",50,low-.2*low,high+.1*high);

    // Fill barrel ET sum histos
    etsumMean_barl_[ieta]=0.;
    for (int iphi=0; iphi<360; iphi++) {
      for (int sign=0; sign<2; sign++) {
	float etsum = etsum_barl_[ieta][iphi][sign];
	etsum_barl_histos_[ieta]->Fill(etsum);
	etsumMean_barl_[ieta]+=etsum;
      }
    }
    etsum_barl_histos_[ieta]->Fit("gaus");
    etsum_barl_histos_[ieta]->Write();
    etsumMean_barl_[ieta]/=720.;
  }

  for (int ring=0; ring<39; ring++) {

    // Determine ranges of ET sums to get histo bounds and book histos (endcap)
    float low=999999.;
    float high=0.;
    for (int ix=0; ix<100; ix++) {
      for (int iy=0; iy<100; iy++) {
	if (endcapRing_[ix][iy]==ring) {
	  for (int sign=0; sign<2; sign++) {
	    float etsum = etsum_endc_[ix][iy][sign];
	    if (etsum<low) low=etsum;
	    if (etsum>high) high=etsum;
	  }
	}
      }
    }
    char ch[3];
    sprintf(ch, "etsum_endc_%i", ring+1);
    etsum_endc_histos_[ring]= new TH1F(ch,"",50,low-.2*low,high+.1*high);

    // Fill endcap ET sum histos
    etsumMean_endc_[ring]=0.;
    for (int ix=0; ix<100; ix++) {
      for (int iy=0; iy<100; iy++) {
	if (endcapRing_[ix][iy]==ring) {
	  for (int sign=0; sign<2; sign++) {
	    float etsum = etsum_endc_[ix][iy][sign];
	    etsum_endc_histos_[ring]->Fill(etsum);
	    etsumMean_endc_[ring]+=etsum;
	  }
	}
      }
    }
    etsum_endc_histos_[ring]->Fit("gaus");
    etsum_endc_histos_[ring]->Write();
    etsumMean_endc_[ring]/=(float(nRing_[ring]*2));
  }

  f.Close();



}
