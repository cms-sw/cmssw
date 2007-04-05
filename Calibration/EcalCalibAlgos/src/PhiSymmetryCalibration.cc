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

//_____________________________________________________________________________

PhiSymmetryCalibration::PhiSymmetryCalibration(const edm::ParameterSet& iConfig) :
  theMaxLoops( iConfig.getUntrackedParameter<unsigned int>("maxLoops",0) ),
  ecalHitsProducer_( iConfig.getParameter< std::string > ("ecalRecHitsProducer") ),
  barrelHits_( iConfig.getParameter< std::string > ("barrelHitCollection") ),
  endcapHits_( iConfig.getParameter< std::string > ("endcapHitCollection") ),
  eCut_barl_( iConfig.getParameter< double > ("eCut_barrel") ),
  eCut_endc_( iConfig.getParameter< double > ("eCut_endcap") )
{

  edm::LogWarning("Calibration") << "[PhiSymmetryCalibration] Constructor called ...";

  theParameterSet=iConfig;

  // Tell the framework what data is being produced
  //setWhatProduced(this);

}


//_____________________________________________________________________________
// Close files, etc.

PhiSymmetryCalibration::~PhiSymmetryCalibration()
{

}

//_____________________________________________________________________________
// Initialize algorithm

void PhiSymmetryCalibration::beginOfJob( const edm::EventSetup& iSetup )
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

  edm::ESHandle<EcalIntercalibConstants> pIcal;
  EcalIntercalibConstants::EcalIntercalibConstantMap imap;

  try {
    iSetup.get<EcalIntercalibConstantsRcd>().get(pIcal);
    std::cout << "Taken EcalIntercalibConstants" << std::endl;
    imap = pIcal.product()->getMap();
    std::cout << "imap.size() = " << imap.size() << std::endl;
  } catch ( std::exception& ex ) {     
    std::cerr << "Error! can't get EcalIntercalibConstants " << std::endl;
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

    // get the initial calibration constants
    EcalIntercalibConstants::EcalIntercalibConstant calib = (imap.find(eb.rawId()))->second;
    int sign = eb.zside()>0 ? 1 : 0;
    oldCalibs_barl[abs(eb.ieta())-1][eb.iphi()-1][sign] = calib;
    if (eb.iphi()==1) std::cout << "Read old constant for crystal "
                           << " (" << eb.ieta() << "," << eb.iphi()
                           << ") : " << calib << std::endl;

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

    // get the initial calibration constants
    EcalIntercalibConstants::EcalIntercalibConstant calib = (imap.find(ee.rawId()))->second;
    int sign = ee.zside()>0 ? 1 : 0;
    oldCalibs_endc[ee.ix()-1][ee.iy()-1][sign] = calib;
    if (ee.ix()==50) std::cout << "Read old constant for crystal "
		        << " (" << ee.ix() << "," << ee.iy()
			       << ") : " << calib << std::endl;

    // store all crystal positions
    cellPos_[ee.ix()-1][ee.iy()-1] = cellGeometry->getPosition();
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

  // calculate eta-phi area for each endcap crystal
  getCellAreas();

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
}

//_____________________________________________________________________________
// Terminate algorithm

void PhiSymmetryCalibration::endOfJob()
{

  edm::LogWarning("Calibration") << "[PhiSymmetryCalibration] At end of job";

  // calculate factors to convert from fractional deviation of ET sum from 
  // the mean to the estimate of the miscalibration factor
  getKfactors();

  // fil output ET sum histograms
  fillHistos();

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
	  float etsum = etsum_endc_[ix][iy][sign]*meanCellArea_[ring]/cellArea_[ix][iy];
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

  std::vector<DetId>::const_iterator barrelIt=barrelCells.begin();
  for (; barrelIt!=barrelCells.end(); barrelIt++) {
    EBDetId eb(*barrelIt);
    int ieta = eb.ieta();
    int iphi = eb.iphi();
    int sign = eb.zside()>0 ? 1 : 0;
    newCalibs_barl[abs(ieta)-1][iphi-1][sign] = 
                                    oldCalibs_barl[abs(ieta)-1][iphi-1][sign]*
                                 (1+epsilon_M_barl[abs(ieta)-1][iphi-1][sign]);
    barrelWriter.writeLine(eb,newCalibs_barl[abs(ieta)-1][iphi-1][sign]);
    if (iphi==1) {
      std::cout << "Calib constant for barrel crystal "
		<< " (" << ieta << "," << iphi << ") changed from "
		<< oldCalibs_barl[abs(ieta)-1][iphi-1][sign] << " to "
		<< newCalibs_barl[abs(ieta)-1][iphi-1][sign] << std::endl;
    }
  }

  std::vector<DetId>::const_iterator endcapIt=endcapCells.begin();
  for (; endcapIt!=endcapCells.end(); endcapIt++) {
    EEDetId ee(*endcapIt);
    int ix = ee.ix();
    int iy = ee.iy();
    int sign = ee.zside()>0 ? 1 : 0;
    newCalibs_endc[ix-1][iy-1][sign] = 
                                    oldCalibs_endc[ix-1][iy-1][sign]*
                                 (1+epsilon_M_endc[ix-1][iy-1][sign]);
    endcapWriter.writeLine(ee,newCalibs_endc[ix-1][iy-1][sign]);
    if (ix==50) {
      std::cout << "Calib constant for endcap crystal "
		<< " (" << ix << "," << iy << "," << sign << ") changed from "
		<< oldCalibs_endc[ix-1][iy-1][sign] << " to "
		<< newCalibs_endc[ix-1][iy-1][sign] << std::endl;
    }
  }

}

//_____________________________________________________________________________
// Called at beginning of loop
void PhiSymmetryCalibration::startingNewLoop(unsigned int iLoop )
{

  edm::LogWarning("Calibration") << "[PhiSymmetryCalibration] Starting loop number " << iLoop;

}


//_____________________________________________________________________________
// Called at end of loop

edm::EDLooper::Status 
PhiSymmetryCalibration::endOfLoop(const edm::EventSetup& iSetup, unsigned int iLoop)
{
  edm::LogWarning("Calibration") << "[PhiSymmetryCalibration] Ending loop " << iLoop;

  if ( iLoop == theMaxLoops-1 || iLoop >= theMaxLoops ) return kStop;
  else return kContinue;
}

//_____________________________________________________________________________
// Called at each event

edm::EDLooper::Status 
PhiSymmetryCalibration::duringLoop( const edm::Event& event, 
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

  try {
    event.getByLabel(ecalHitsProducer_,barrelHits_,barrelRecHitsHandle);
    event.getByLabel(ecalHitsProducer_,endcapHits_,endcapRecHitsHandle);
  } catch ( std::exception& ex ) {
    LogDebug("") << "PhiSymmetryCalibration: Error! can't get product!" << std::endl;
  }

  //Select interesting EcalRecHits (barrel)
  EBRecHitCollection::const_iterator itb;
  for (itb=barrelRecHitsHandle->begin(); itb!=barrelRecHitsHandle->end(); itb++) {
    EBDetId hit = EBDetId(itb->id());
    float eta = cellEta_[abs(hit.ieta())-1];
    float et = itb->energy()/cosh(eta);
    float et_thr = eCut_barl_/cosh(eta);
    if (et > et_thr && et < et_thr+0.8) {
      int sign = hit.ieta()>0 ? 1 : 0;
      etsum_barl_[abs(hit.ieta())-1][hit.iphi()-1][sign] += et;
    }

    //Apply a miscalibration to all crystals and increment the 
    //ET sum, combined for all crystals
    for (int imiscal=0; imiscal<21; imiscal++) {
      if (miscal_[imiscal]*et > et_thr && miscal_[imiscal]*et < et_thr+0.8) {
	etsum_barl_miscal_[imiscal][abs(hit.ieta())-1] += miscal_[imiscal]*et;
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
    if (et > et_thr && et < et_thr+0.8) {
      int sign = hit.zside()>0 ? 1 : 0;
      etsum_endc_[hit.ix()-1][hit.iy()-1][sign] += et;
    }

    //Apply a miscalibration to all crystals and increment the 
    //ET sum, combined for all crystals
    for (int imiscal=0; imiscal<21; imiscal++) {
      if (miscal_[imiscal]*et > et_thr && miscal_[imiscal]*et < et_thr+0.8) {
	int ring = endcapRing_[hit.ix()-1][hit.iy()-1];
	etsum_endc_miscal_[imiscal][ring] += miscal_[imiscal]*et*meanCellArea_[ring]/cellArea_[hit.ix()-1][hit.iy()-1];
      }
    }

  }

  return kContinue;
}

// ----------------------------------------------------------------------------

void PhiSymmetryCalibration::getCellAreas()
{

  for (int ix=0; ix<100; ix++) {
    for (int iy=0; iy<100; iy++) {
      if (cellPos_[ix][iy].x()!=0. && cellPos_[ix][iy].y()!=0.) {

	float xcoord[4],ycoord[4],etacoord[4],phicoord[4];
	for (int index=0; index<4; index++) {
	  xcoord[index]=0.;
	  ycoord[index]=0.;
	  etacoord[index]=0.;
	  phicoord[index]=0.;
	}

	if (ix!=0 && cellPos_[ix-1][iy].x()!=0.) {
	  xcoord[0]=(cellPos_[ix][iy].x()+cellPos_[ix-1][iy].x())/2.;
	} else {
	  xcoord[0]=cellPos_[ix][iy].x()-(cellPos_[ix+1][iy].x()-cellPos_[ix][iy].x())/2.;
	}
	if (iy!=0 && iy!=50 && cellPos_[ix][iy-1].y()!=0.) {
	  ycoord[0]=(cellPos_[ix][iy].y()+cellPos_[ix][iy-1].y())/2.;
	} else {
	  ycoord[0]=cellPos_[ix][iy].y()-(cellPos_[ix][iy+1].y()-cellPos_[ix][iy].y())/2.;
	}
	if (ix!=99 && cellPos_[ix+1][iy].x()!=0.) {
	  xcoord[2]=(cellPos_[ix][iy].x()+cellPos_[ix+1][iy].x())/2.;
	} else {
	  xcoord[2]=cellPos_[ix][iy].x()+(cellPos_[ix][iy].x()-cellPos_[ix-1][iy].x())/2.;
	}
	if (iy!=99 && iy!=49 && cellPos_[ix][iy+1].y()!=0.) {
	  ycoord[2]=(cellPos_[ix][iy].y()+cellPos_[ix][iy+1].y())/2.;
	} else {
	  ycoord[2]=cellPos_[ix][iy].y()+(cellPos_[ix][iy].y()-cellPos_[ix][iy-1].y())/2.;
	}
	xcoord[1]=xcoord[2];
	xcoord[3]=xcoord[0];
	ycoord[1]=ycoord[0];
	ycoord[3]=ycoord[2];

	for (int index=0; index<4; index++) {
	  phicoord[index] = atan(xcoord[index]/ycoord[index]);
	  float theta = atan(sqrt(xcoord[index]*xcoord[index] +
			          ycoord[index]*ycoord[index])/320.5);
	  etacoord[index] = -log(tan(theta/2.));
	}

	cellArea_[ix][iy]=0.;
	for (int index=0; index<4; index++) {
	  int indexplus1 = index==3 ? 0 : index+1;
	  cellArea_[ix][iy] += etacoord[index]*phicoord[indexplus1] - 
	                       etacoord[indexplus1]*phicoord[index];
	}
	cellArea_[ix][iy] = cellArea_[ix][iy]/2.;

      }
    }
  }
}


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
    etsum_barl_histos_[ieta] = new TH1F(ch,"",50,low-.2*low,high+.1*high);

    // Fill barrel ET sum histos
    etsumMean_barl_[ieta]=0.;
    for (int iphi=0; iphi<360; iphi++) {
      for (int sign=0; sign<2; sign++) {
	float etsum = etsum_barl_[ieta][iphi][sign];
	etsum_barl_histos_[ieta]->Fill(etsum);
	etsumMean_barl_[ieta]+=etsum;
	//std::cout << ieta << " " << iphi << " " << etsum_barl_[ieta][iphi][sign] << endl;
      }
    }
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
    etsum_endc_histos_[ring] = new TH1F(ch,"",50,low-.2*low,high+.1*high);

    // Fill endcap ET sum histos
    etsumMean_endc_[ring]=0.;
    for (int ix=0; ix<100; ix++) {
      for (int iy=0; iy<100; iy++) {
	if (endcapRing_[ix][iy]==ring) {
	  for (int sign=0; sign<2; sign++) {
	    float etsum = etsum_endc_[ix][iy][sign];
	    etsum_endc_histos_[ring]->Fill(
		etsum*(meanCellArea_[ring]/cellArea_[ix][iy]));
	    etsumMean_endc_[ring] += 
	        etsum*(meanCellArea_[ring]/cellArea_[ix][iy]);
	  }
	}
      }
    }
    etsum_endc_histos_[ring]->Write();
    etsumMean_endc_[ring]/=(float(nRing_[ring]*2));
  }

  for (int ieta=0; ieta<85; ieta++) k_barl_plot_[ieta]->Write();
  for (int ring=0; ring<39; ring++) k_endc_plot_[ring]->Write();

  f.Close();

}
