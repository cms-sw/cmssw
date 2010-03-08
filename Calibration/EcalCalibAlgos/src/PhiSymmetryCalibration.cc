#include "Calibration/EcalCalibAlgos/interface/PhiSymmetryCalibration.h"

// System include files
#include <memory>

// Framework
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"

// Conditions database

#include "CondTools/Ecal/interface/EcalIntercalibConstantsXMLTranslator.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibErrors.h"


#include "Calibration/Tools/interface/calibXMLwriter.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/MiscalibReaderFromXMLEcalBarrel.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/MiscalibReaderFromXMLEcalEndcap.h"

// Geometry
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Calibration/Tools/interface/EcalRingCalibrationTools.h"

//Channel status
#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatusCode.h"


using namespace std;
#include <fstream>
#include <iostream>
#include <float.h>
#include "TH2F.h"

const float PhiSymmetryCalibration::kMiscalRangeEB = .05;
const float PhiSymmetryCalibration::kMiscalRangeEE = .15;

//E_T spactrum histos
std::vector<TH1F*> et_spectrum_b_histos(85); //kBarlEtaRings
std::vector<TH1F*> e_spectrum_b_histos(85);
std::vector<TH1F*> et_spectrum_e_histos(39); //kEndcEtaRings
std::vector<TH1F*> e_spectrum_e_histos(39);

std::vector<TH1F*> et_crystal_b_histos(360);
std::vector<TH1F*> e_crystal_b_histos(360);
std::vector<TH1F*> et_crystal_e_histos(220);
std::vector<TH1F*> e_crystal_e_histos(220);

Bool_t spectra = true;



//_____________________________________________________________________________
// Class constructor

PhiSymmetryCalibration::PhiSymmetryCalibration(const edm::ParameterSet& iConfig) :

  ecalHitsProducer_( iConfig.getParameter< std::string > ("ecalRecHitsProducer") ),
  barrelHits_( iConfig.getParameter< std::string > ("barrelHitCollection") ),
  endcapHits_( iConfig.getParameter< std::string > ("endcapHitCollection") ),
  eCut_barl_( iConfig.getParameter< double > ("eCut_barrel") ),
  eCut_endc_( iConfig.getParameter< double > ("eCut_endcap") ),
  ap_( iConfig.getParameter< float > ("ap") ),
  am_( iConfig.getParameter< float > ("am") ),
  b_( iConfig.getParameter< float > ("b") ), 

  eventSet_( iConfig.getParameter< int > ("eventSet") ),
  statusThreshold_(iConfig.getUntrackedParameter<int>("statusThreshold",1000)),
  reiteration_(iConfig.getUntrackedParameter< bool > ("reiteration",false)),
  reiterationXMLFileEB_(iConfig.getUntrackedParameter< std::string > ("reiterationXMLFileEB","")),
  reiterationXMLFileEE_(iConfig.getUntrackedParameter< std::string > ("reiterationXMLFileEE",""))

{

  newCalibs_.prefillMap();
  isfirstpass_=true;
}


//_____________________________________________________________________________
// Close files, etc.

PhiSymmetryCalibration::~PhiSymmetryCalibration()
{

}


//_____________________________________________________________________________
// Initialize algorithm

void PhiSymmetryCalibration::beginJob( )
{

  edm::LogInfo("Calibration") << "[PhiSymmetryCalibration] At begin job ...";

  // initialize arrays
  for (int sign=0; sign<kSides; sign++) {
    for (int ieta=0; ieta<kBarlRings; ieta++) {
      for (int iphi=0; iphi<kBarlWedges; iphi++) {
	etsum_barl_[ieta][iphi][sign]=0.;
	nhits_barl_[ieta][iphi][sign]=0;
	esum_barl_[ieta][iphi][sign]=0.;
	eta_barl_[ieta][iphi][sign]=0.;
	phi_barl_[ieta][iphi][sign]=0.;

	goodCell_barl[ieta][iphi][sign] = false;
      }
    }
    for (int ix=0; ix<kEndcWedgesX; ix++) {
      for (int iy=0; iy<kEndcWedgesY; iy++) {
	etsum_endc_[ix][iy][sign]=0.;
	nhits_endc_[ix][iy][sign]=0;
	esum_endc_[ix][iy][sign]=0.;
	eta_endc_[ix][iy][sign]=0.;
	phi_endc_[ix][iy][sign]=0.;

	goodCell_endc[ix][iy][sign] = false;
      }
    }
  }

  for (int ieta=0; ieta<kBarlRings; ieta++) cellEta_[ieta] = 0.;

  for (int ieta=0; ieta<kBarlRings; ieta++)    nBads_barl[ieta] = 0;
  for (int ring=0; ring<kEndcEtaRings; ring++) nBads_endc[ring] = 0;

  for (int ix=0; ix<kEndcWedgesX; ix++) {
    for (int iy=0; iy<kEndcWedgesY; iy++) {
      cellPos_[ix][iy] = GlobalPoint(0.,0.,0.);
      cellPhi_[ix][iy]=0.;
      cellArea_[ix][iy]=0.;
      endcapRing_[ix][iy]=-1;
    }
  }

  for (int imiscal=0; imiscal<kNMiscalBinsEB; imiscal++) {
    miscalEB_[imiscal]= (1-kMiscalRangeEB) + float(imiscal)* (2*kMiscalRangeEB/(kNMiscalBinsEB-1));
    for (int ieta=0; ieta<kBarlRings; ieta++) etsum_barl_miscal_[imiscal][ieta]=0.;
  }

  for (int imiscal=0; imiscal<kNMiscalBinsEE; imiscal++) {
    miscalEE_[imiscal]= (1-kMiscalRangeEE) + float(imiscal)* (2*kMiscalRangeEE/(kNMiscalBinsEE-1));
    for (int ring=0; ring<kEndcEtaRings; ring++) etsum_endc_miscal_[imiscal][ring]=0.;
  }


 


  // start spectra stuff
  if (eventSet_!=1) spectra = false;
  
  if(spectra)
    {
      ostringstream t;
      for(Int_t i=0;i<kBarlRings;i++)
	{
	  t << "et_spectrum_b_" << i+1;
	  et_spectrum_b_histos[i]=new TH1F(t.str().c_str(),";E_{T} [MeV]",50,0.,500.);
	  t.str("");
	  
	  t << "e_spectrum_b_" << i+1;
	  e_spectrum_b_histos[i]=new TH1F(t.str().c_str(),";E [MeV]",50,0.,500.);
	  t.str("");
	  /*
	    t << "et_crystal_b_" << i+1 << "_110";
	    et_crystal_b_histos[i]=new TH1F(t.str().c_str(),";E_{T} [MeV]",50,0.,1000.);
	    t.str("");
	    
	    t << "e_crystal_b_" << i+1 << "_110";
	    e_crystal_b_histos[i]=new TH1F(t.str().c_str(),";E [MeV]",50,0.,1000.);
	    t.str("");
	  */
	}
      for(Int_t i=0;i<kBarlWedges;i++)
	{
	  t << "et_crystal_b_21_" << i+1;
	  et_crystal_b_histos[i]=new TH1F(t.str().c_str(),";E_{T} [MeV]",50,0.,1000.);
	  t.str("");
	  
	  t << "e_crystal_b_21_" << i+1;
	  e_crystal_b_histos[i]=new TH1F(t.str().c_str(),";E [MeV]",50,0.,1000.);
	  t.str("");
	}
      for(Int_t i=0;i<kEndcEtaRings;i++)
	{
	  t << "et_spectrum_e_" << i+1;
	  et_spectrum_e_histos[i]=new TH1F(t.str().c_str(),";E_{T} [MeV]",75,0.,1500.);
	  t.str("");
	  
	  t << "e_spectrum_e_" << i+1;
	  e_spectrum_e_histos[i]=new TH1F(t.str().c_str(),";E [MeV]",75,0.,1500.);
	  t.str("");
	  /*	    
	    t << "et_crystal_e_" << i+1 << "_51";
	    et_crystal_e_histos[i]=new TH1F(t.str().c_str(),";E_{T} [MeV]",75,0.,1500.);
	    t.str("");
	    
	    t << "e_crystal_e_" << i+1 << "_51";
	    e_crystal_e_histos[i]=new TH1F(t.str().c_str(),";E [MeV]",75,0.,1500.);
	    t.str("");	    
	  */
	}
      for(Int_t i=0;i<nRing_[16];i++)
	{
	  t << "et_crystal_e_17_" << i+1;
	  et_crystal_e_histos[i]=new TH1F(t.str().c_str(),";E_{T} [MeV]",100,0.,2000.);
	  t.str("");
	  
	  t << "e_crystal_e_17_" << i+1;
	  e_crystal_e_histos[i]=new TH1F(t.str().c_str(),";E [MeV]",100,0.,2000.);
	  t.str("");	    
	}
    }
  // end spectra stuff
}


//_____________________________________________________________________________
// Terminate algorithm

void PhiSymmetryCalibration::endJob()
{

  edm::LogWarning("Calibration") << "[PhiSymmetryCalibration] At end of job";

  // start spectra stuff
  if(spectra)
    {
      TFile f("Espectra_plus.root","recreate");
      //	TFile f("Espectra_minus.root","recreate");
      for(Int_t i=0;i<kBarlRings;i++){
	et_spectrum_b_histos[i]->Write();
	e_spectrum_b_histos[i]->Write();
	//	  et_crystal_b_histos[i]->Write();
	//	  e_crystal_b_histos[i]->Write();
      }
      for(Int_t i=0;i<kBarlWedges;i++){
	et_crystal_b_histos[i]->Write();
	e_crystal_b_histos[i]->Write();
      }
      for(Int_t i=0;i<kEndcEtaRings;i++){
	et_spectrum_e_histos[i]->Write();
	e_spectrum_e_histos[i]->Write();
	//	  et_crystal_e_histos[i]->Write();
	//	  e_crystal_e_histos[i]->Write();
      }
      for(Int_t i=0;i<nRing_[16];i++){
	et_crystal_e_histos[i]->Write();
	e_crystal_e_histos[i]->Write();
      }
      f.Close();
    }
  
    for(Int_t i=0;i<kBarlRings;i++){
    delete et_spectrum_b_histos[i];
    delete e_spectrum_b_histos[i];
    delete et_crystal_b_histos[i];
    delete e_crystal_b_histos[i];
  }
  for(Int_t i=0;i<kEndcEtaRings;i++){
    delete et_spectrum_e_histos[i];
    delete e_spectrum_e_histos[i];
    delete et_crystal_e_histos[i];
    delete e_crystal_e_histos[i];
  }
  // end sprectra stuff


  if (eventSet_==1) {
    // calculate factors to convert from fractional deviation of ET sum from 
    // the mean to the estimate of the miscalibration factor
    getKfactors();

    std::ofstream k_barl_out("k_barl.dat", ios::out);
    for (int ieta=0; ieta<kBarlRings; ieta++)
      k_barl_out << ieta << " " << k_barl_[ieta] << endl;
    k_barl_out.close();

    std::ofstream k_endc_out("k_endc.dat", ios::out);
    for (int ring=0; ring<kEndcEtaRings; ring++)
      k_endc_out << ring << " " << k_endc_[ring] << endl;
    k_endc_out.close();
  }


  if (eventSet_!=0) {
    // output ET sums

    stringstream etsum_file_barl;
    etsum_file_barl << "etsum_barl_"<<eventSet_<<".dat";

    std::ofstream etsum_barl_out(etsum_file_barl.str().c_str(),ios::out);

    for (int ieta=0; ieta<kBarlRings; ieta++) {
      for (int iphi=0; iphi<kBarlWedges; iphi++) {
	for (int sign=0; sign<kSides; sign++) {
	  etsum_barl_out << eventSet_ << " " << ieta << " " << iphi << " " << sign 
			 << " " << etsum_barl_[ieta][iphi][sign] << " " <<  nhits_barl_[ieta][iphi][sign]
			 << " " << esum_barl_[ieta][iphi][sign]  << " " <<  eta_barl_[ieta][iphi][sign] 
			 << " " << phi_barl_[ieta][iphi][sign]   << endl;
	}
      }
    }
    etsum_barl_out.close();

    stringstream etsum_file_endc;
    etsum_file_endc << "etsum_endc_"<<eventSet_<<".dat";

    std::ofstream etsum_endc_out(etsum_file_endc.str().c_str(),ios::out);
    for (int ix=0; ix<kEndcWedgesX; ix++) {
      for (int iy=0; iy<kEndcWedgesY; iy++) {
	int ring = endcapRing_[ix][iy];
	if (ring!=-1) {
	  for (int sign=0; sign<kSides; sign++) {
	    etsum_endc_out << eventSet_ << " " << ix << " " << iy << " " << sign 
			   << " " << etsum_endc_[ix][iy][sign] << " " << nhits_endc_[ix][iy][sign]
			   << " " << esum_endc_[ix][iy][sign]  << " " << eta_endc_[ix][iy][sign]
			   << " " << phi_endc_[ix][iy][sign]   << endl;
	  }
	}
      }
    }
    etsum_endc_out.close();

  } else {   // eventSet_ =0

    for (int ieta=0; ieta<kBarlRings; ieta++) {
      for (int iphi=0; iphi<kBarlWedges; iphi++) {
	for (int sign=0; sign<kSides; sign++) {
	  etsum_barl_[ieta][iphi][sign]=0.;
	  nhits_barl_[ieta][iphi][sign]=0;
	  esum_barl_[ieta][iphi][sign]=0.;
	  eta_barl_[ieta][iphi][sign]=0.;
	  phi_barl_[ieta][iphi][sign]=0.;
	}
      }
    }

    for (int ix=0; ix<kEndcWedgesX; ix++) {
      for (int iy=0; iy<kEndcWedgesY; iy++) {
	for (int sign=0; sign<kSides; sign++) {
	  etsum_endc_[ix][iy][sign]=0.;
	  nhits_endc_[ix][iy][sign]=0;
	  esum_endc_[ix][iy][sign]=0.;
	  eta_endc_[ix][iy][sign]=0.;
	  phi_endc_[ix][iy][sign]=0.;
	}
      }
    }


    //read in ET sums

    int ieta,iphi,sign,ix,iy,dummy;
    double etsum, esum, eta, phi;
    unsigned int nhits;
    std::ifstream etsum_barl_in("etsum_barl.dat", ios::in);
    while ( etsum_barl_in >> dummy >> ieta >> iphi >> sign >> etsum >> nhits >> esum >> eta >> phi ) {
      etsum_barl_[ieta][iphi][sign]+=etsum;
      nhits_barl_[ieta][iphi][sign]+=nhits;
      esum_barl_[ieta][iphi][sign]+=esum;
      eta_barl_[ieta][iphi][sign]=eta;
      phi_barl_[ieta][iphi][sign]=phi;
    }
    etsum_barl_in.close();

    std::ifstream etsum_endc_in("etsum_endc.dat", ios::in);
    while ( etsum_endc_in >> dummy >> ix >> iy >> sign >> etsum >> nhits >> esum >> eta >> phi ) {
      etsum_endc_[ix][iy][sign]+=etsum;
      nhits_endc_[ix][iy][sign]+=nhits;
      esum_endc_[ix][iy][sign]+=esum;
      eta_endc_[ix][iy][sign]=eta;
      phi_endc_[ix][iy][sign]=phi;
    }
    etsum_endc_in.close();

    std::ifstream k_barl_in("k_barl.dat", ios::in);
    for (int ieta=0; ieta<kBarlRings; ieta++) {
      k_barl_in >> dummy >> k_barl_[ieta];
    }
    k_barl_in.close();

    std::ifstream k_endc_in("k_endc.dat", ios::in);
    for (int ring=0; ring<kEndcEtaRings; ring++) {
      k_endc_in >> dummy >> k_endc_[ring];
    }
    k_endc_in.close();


    // perform the area correction for endcap etsum
    // NO MORE USED
    for (int ix=0; ix<kEndcWedgesX; ix++) {
      for (int iy=0; iy<kEndcWedgesY; iy++) {

        int ring = endcapRing_[ix][iy];

        if (ring!=-1) {
          for (int sign=0; sign<kSides; sign++) {
            etsum_endc_uncorr[ix][iy][sign] = etsum_endc_[ix][iy][sign];
	    //            etsum_endc_[ix][iy][sign]*=meanCellArea_[ring]/cellArea_[ix][iy];
          }
        }
      }
    }


    // ETsum histos, maps and other usefull histos (area,...)
    // are filled and saved here
    fillHistos();


    // write ETsum mean for all rings
    std::ofstream etsumMean_barl_out("etsumMean_barl.dat",ios::out);
    for (int ieta=0; ieta<kBarlRings; ieta++) {
      etsumMean_barl_out << cellEta_[ieta] << " " << etsumMean_barl_[ieta] << endl;
    }
    etsumMean_barl_out.close();

    std::ofstream etsumMean_endc_out("etsumMean_endc.dat",ios::out);
    for (int ring=0; ring<kEndcEtaRings; ring++) {
      etsumMean_endc_out << cellPos_[ring][50].eta() << " " << etsumMean_endc_[ring] << endl;
    }
    etsumMean_endc_out.close();

    std::ofstream area_out("area.dat",ios::out);
    for (int ring=0; ring<kEndcEtaRings; ring++) {
      area_out << meanCellArea_[ring] << endl;
    }
    area_out.close();


    // determine barrel calibration constants
    for (int ieta=0; ieta<kBarlRings; ieta++) {
      for (int iphi=0; iphi<kBarlWedges; iphi++) {
	for (int sign=0; sign<kSides; sign++) {
	  if(goodCell_barl[ieta][iphi][sign]){
	  float etsum = etsum_barl_[ieta][iphi][sign];
	  float epsilon_T = (etsum/etsumMean_barl_[ieta]) - 1.;
	  rawconst_barl[ieta][iphi][sign]  = epsilon_T + 1.;
	  epsilon_M_barl[ieta][iphi][sign] = epsilon_T/k_barl_[ieta];
	  } else {
	  rawconst_barl[ieta][iphi][sign]  = 1.;
	  epsilon_M_barl[ieta][iphi][sign] = 0.;
	  } //if
	} //sign
      } //iphi
    } //ieta

    // determine endcap calibration constants
    for (int ix=0; ix<kEndcWedgesX; ix++) {
      for (int iy=0; iy<kEndcWedgesY; iy++) {
	for (int sign=0; sign<kSides; sign++) {
	  int ring = endcapRing_[ix][iy];
	  if (ring!=-1 && goodCell_endc[ix][iy][sign]) {
	    float etsum = etsum_endc_[ix][iy][sign];
	    float epsilon_T = (etsum/etsumMean_endc_[ring]) - 1.;
	    rawconst_endc[ix][iy][sign]  = epsilon_T + 1.;
	    epsilon_M_endc[ix][iy][sign] = epsilon_T/k_endc_[ring];	    
	  } else {
	    epsilon_M_endc[ix][iy][0] = 0.;
	    epsilon_M_endc[ix][iy][1] = 0.;
	    rawconst_endc[ix][iy][0]  = 1.;
	    rawconst_endc[ix][iy][1]  = 1.;
	  } //if
	} //sign
      } //iy
    } //ix


    // write new calibration constants
    calibXMLwriter barrelWriter(EcalBarrel);
    calibXMLwriter endcapWriter(EcalEndcap);

    // records to be filled   
    EcalIntercalibConstants intercalib_constants;
    EcalIntercalibErrors    intercalib_errors;

    std::string newcalibfile("EcalIntercalibConstants.xml");



    std::vector<TH1F*> miscal_resid_barl_histos(kBarlRings);
    std::vector<TH2F*> correl_barl_histos(kBarlRings);  
 
    std::ofstream calibs_barl_out("calibs_barl.dat",ios::out);

     for (int ieta=0; ieta<kBarlRings; ieta++) {
       ostringstream t1; 
       t1<<"mr_barl_"<<ieta+1;
       miscal_resid_barl_histos[ieta] = new TH1F(t1.str().c_str(),"",100,0.,2.);
       ostringstream t2;
       t2<<"co_barl_"<<ieta+1;
       correl_barl_histos[ieta] = new TH2F(t2.str().c_str(),"",50,.5,1.5,50,.5,1.5);
     }

    std::vector<TH1F*> miscal_resid_endc_histos(kEndcEtaRings);
    std::vector<TH2F*> correl_endc_histos(kEndcEtaRings);

    std::ofstream calibs_endc_out("calibs_endc.dat",ios::out);

    for (int ring=0; ring<kEndcEtaRings; ring++) {
      ostringstream t1;
      t1<<"mr_endc_"<< ring+1;
      miscal_resid_endc_histos[ring] = new TH1F(t1.str().c_str(),"",100,0.,2.);
      ostringstream t2;
      t2<<"co_endc_"<<ring+1;
      correl_endc_histos[ring] = new TH2F(t2.str().c_str(),"",50,.5,1.5,50,.5,1.5);
    }


    TFile ehistof("ehistos.root","recreate");  

    TH1D ebhisto("eb","eb",100, 0.,2.);
    std::vector<DetId>::const_iterator barrelIt=barrelCells.begin();
    for (; barrelIt!=barrelCells.end(); barrelIt++) {
      EBDetId eb(*barrelIt);
      int ieta = abs(eb.ieta())-1;
      int iphi = eb.iphi()-1;
      int sign = eb.zside()>0 ? 1 : 0;

      newCalibs_barl[ieta][iphi][sign] =  1./(1+epsilon_M_barl[ieta][iphi][sign]);

      barrelWriter.writeLine(eb,newCalibs_barl[ieta][iphi][sign]);

      if(goodCell_barl[ieta][iphi][sign]){
	ebhisto.Fill(newCalibs_barl[ieta][iphi][sign]);
	miscal_resid_barl_histos[ieta]->Fill(1/(1+epsilon_M_barl[ieta][iphi][sign]));
	correl_barl_histos[ieta]->Fill(oldCalibs_barl[ieta][iphi][sign],newCalibs_barl[ieta][iphi][sign]);
	
	// fill record
	intercalib_constants[eb]=newCalibs_barl[ieta][iphi][sign];
	intercalib_constants[eb]=1.;      
	
	miscal_resid_barl_histos[ieta]->Fill(newCalibs_barl[ieta][iphi][sign]);
	correl_barl_histos[ieta]->Fill(oldCalibs_barl[ieta][iphi][sign],1+epsilon_M_barl[ieta][iphi][sign]);
	if (iphi==1) {
	  std::cout << "Calib constant for barrel crystal "
		    << " (" << eb.ieta() << "," << eb.iphi() << ") changed from "
		    << oldCalibs_barl[ieta][iphi][sign] << " to "
		    << newCalibs_barl[ieta][iphi][sign] << std::endl;
	  
	}
	
	calibs_barl_out << ieta << " " << iphi << " " << sign << " "
			<< oldCalibs_barl[ieta][iphi][sign]   << " "
			<< newCalibs_barl[ieta][iphi][sign] <<  std::endl;
      }
      
    }// barrelit

    TH1D eehisto("ee","ee",100, 0.,2.);
    std::vector<DetId>::const_iterator endcapIt=endcapCells.begin();
    for (; endcapIt!=endcapCells.end(); endcapIt++) {
      EEDetId ee(*endcapIt);
      int ix = ee.ix()-1;
      int iy = ee.iy()-1;
      int sign = ee.zside()>0 ? 1 : 0;
      
      newCalibs_endc[ix][iy][sign] = 1./(1+epsilon_M_endc[ix][iy][sign]);
      
      endcapWriter.writeLine(ee,newCalibs_endc[ix][iy][sign]);
      
      if(goodCell_endc[ix][iy][sign]){ 
	eehisto.Fill(newCalibs_endc[ix][iy][sign]);
	miscal_resid_endc_histos[endcapRing_[ix][iy]]->Fill(1/(1+epsilon_M_endc[ix][iy][sign]));
	correl_endc_histos[endcapRing_[ix][iy]]->Fill(oldCalibs_endc[ix][iy][sign],newCalibs_endc[ix][iy][sign]);
	// fill record
	intercalib_constants[ee]=newCalibs_endc[ix][iy][sign];
	intercalib_constants[ee]=1.;

	miscal_resid_endc_histos[endcapRing_[ix][iy]]->Fill(newCalibs_endc[ix][iy][sign]);
	correl_endc_histos[endcapRing_[ix][iy]]->Fill(oldCalibs_endc[ix][iy][sign],1+epsilon_M_endc[ix][iy][sign]);
	if (ix==50) {
	  std::cout << "Calib constant for endcap crystal "
		    << " (" << ix << "," << iy << "," << sign << ") changed from "
		    << oldCalibs_endc[ix][iy][sign] << " to "
		    << newCalibs_endc[ix][iy][sign] << std::endl;

	}

	calibs_endc_out << endcapRing_[ix][iy]  << " "  << ix << " " << iy << " " << sign << " "
			<< oldCalibs_endc[ix][iy][sign] <<  " " 
			<< newCalibs_endc[ix][iy][sign] << std::endl;
      }
    }//endcapit

    // Write xml file
    EcalCondHeader header;
    header.method_="phi symmetry";
    header.version_="0";
    header.datasource_="testdata";
    header.since_=1;
    header.tag_="unknown";
    header.date_="Mar 24 1973";

    EcalIntercalibConstantsXMLTranslator::writeXML(newcalibfile,header,
						   intercalib_constants );


    eehisto.Write();
    ebhisto.Write();
    ehistof.Close();

    calibs_barl_out.close();
    calibs_endc_out.close();


    // output histograms of residual miscalibrations
    TFile f("PhiSymmetryCalibration_miscal_resid.root","recreate");
    for (int ieta=0; ieta<85; ieta++) {
      miscal_resid_barl_histos[ieta]->Write();
      correl_barl_histos[ieta]->Write();

      delete miscal_resid_barl_histos[ieta];
      delete correl_barl_histos[ieta];
    }
    for (int ring=0; ring<39; ring++) {
      miscal_resid_endc_histos[ring]->Write();
      correl_endc_histos[ring]->Write();

      delete  miscal_resid_endc_histos[ring];
      delete  correl_endc_histos[ring];

    }
    f.Close();
    fillConstantsHistos();
    
  }// else (if eventset=0)
}


//_____________________________________________________________________________
// Called at each event

void PhiSymmetryCalibration::analyze( const edm::Event& event, const edm::EventSetup& setup )
{
  using namespace edm;
  using namespace std;
  
  if (isfirstpass_) {
    setUp(setup);
    isfirstpass_=false;
  }
   
  
  Handle<EBRecHitCollection> barrelRecHitsHandle;
  Handle<EERecHitCollection> endcapRecHitsHandle;
  
  event.getByLabel(ecalHitsProducer_,barrelHits_,barrelRecHitsHandle);
  if (!barrelRecHitsHandle.isValid()) {
    LogDebug("") << "[PhiSymmetryCalibration] Error! Can't get product!" << std::endl;
  }
  
  event.getByLabel(ecalHitsProducer_,endcapHits_,endcapRecHitsHandle);
  if (!endcapRecHitsHandle.isValid()) {
    LogDebug("") << "[PhiSymmetryCalibration] Error! Can't get product!" << std::endl;
  }
  
  // select interesting EcalRecHits (barrel)
  EBRecHitCollection::const_iterator itb;
  for (itb=barrelRecHitsHandle->begin(); itb!=barrelRecHitsHandle->end(); itb++) {
    EBDetId hit = EBDetId(itb->id());
    float eta = cellEta_[abs(hit.ieta())-1];
    float phi = cellPhiB_[hit.iphi()-1]; //VS
    float et = itb->energy()/cosh(eta);
    float e  = itb->energy();
    
    // if (reiteration_) et= et /  previousCalibs_[hit] * newCalibs_.get()[hit];

    float et_thr = eCut_barl_/cosh(eta) + 1.;

    int sign = hit.ieta()>0 ? 1 : 0;
    if (e >  eCut_barl_ && et < et_thr && goodCell_barl[abs(hit.ieta())-1][hit.iphi()-1][sign]) {
      etsum_barl_[abs(hit.ieta())-1][hit.iphi()-1][sign] += et;
      nhits_barl_[abs(hit.ieta())-1][hit.iphi()-1][sign] ++;
      esum_barl_[abs(hit.ieta())-1][hit.iphi()-1][sign] += e;
    }//if energy
    eta_barl_[abs(hit.ieta())-1][hit.iphi()-1][sign]  = eta;
    phi_barl_[abs(hit.ieta())-1][hit.iphi()-1][sign]  = phi;

    if (eventSet_==1) {
      // apply a miscalibration to all crystals and increment the 
      // ET sum, combined for all crystals
      for (int imiscal=0; imiscal<kNMiscalBinsEB; imiscal++) {
	if (miscalEB_[imiscal]*e >  eCut_barl_&& miscalEB_[imiscal]*et < et_thr && goodCell_barl[abs(hit.ieta())-1][hit.iphi()-1][sign]) {
	  etsum_barl_miscal_[imiscal][abs(hit.ieta())-1] += miscalEB_[imiscal]*et;
	}
      }

      // spectra stuff
      if(spectra && hit.ieta()>0) //POSITIVE!!!
	//      if(spectra && hit.ieta()<0) //NEGATIVE!!!
	{
	  et_spectrum_b_histos[abs(hit.ieta())-1]->Fill(et*1000.);
	  e_spectrum_b_histos[abs(hit.ieta())-1]->Fill(e*1000.);
	  /*	  
	  if(hit.iphi()==110)
	    {
	      et_crystal_b_histos[abs(hit.ieta())-1]->Fill(et*1000.);
	      e_crystal_b_histos[abs(hit.ieta())-1]->Fill(e*1000.);
	    }
	  */
	  if(hit.ieta()==20)
	    {
	      et_crystal_b_histos[hit.iphi()-1]->Fill(et*1000.);
	      e_crystal_b_histos[hit.iphi()-1]->Fill(e*1000.);
	    }
	}//if spectra
      
    }//if eventSet_==1
  }//for barl


  // select interesting EcalRecHits (endcaps)
  EERecHitCollection::const_iterator ite;
  for (ite=endcapRecHitsHandle->begin(); ite!=endcapRecHitsHandle->end(); ite++) {
    EEDetId hit = EEDetId(ite->id());
    float eta = cellPos_[hit.ix()-1][hit.iy()-1].eta();
    float phi = cellPos_[hit.ix()-1][hit.iy()-1].phi(); //VS
    float et = ite->energy()/cosh(eta);
    float e  = ite->energy();

    int sign = hit.zside()>0 ? 1 : 0;

    // if (reiteration_) et= et /  previousCalibs_[hit] * newCalibs_.get()[hit];

    // changes of eCut_endc_ -> variable linearthr 
    // EE+ : e_cut = ap + eta_ring*b
    // EE- : e_cut = ap + eta_ring*b
    //
    for (int ring=0; ring<kEndcEtaRings; ring++) {
      if(eta>etaBoundary_[ring] && eta<etaBoundary_[ring+1])
	{
	  float eta_ring=cellPos_[ring][50].eta();
  
	  if(sign==1)
	    eCut_endc_ = ap_ + eta_ring*b_;
	  else
	    eCut_endc_ = am_ + eta_ring*b_;
	}
    }

    float et_thr = eCut_endc_/cosh(eta) + 1.;

    if (e > eCut_endc_ && et < et_thr && goodCell_endc[hit.ix()-1][hit.iy()-1][sign]){
      etsum_endc_[hit.ix()-1][hit.iy()-1][sign] += et;
      nhits_endc_[hit.ix()-1][hit.iy()-1][sign] ++;
      esum_endc_[hit.ix()-1][hit.iy()-1][sign] += e;
    }
    eta_endc_[hit.ix()-1][hit.iy()-1][sign]  = eta;
    phi_endc_[hit.ix()-1][hit.iy()-1][sign]  = phi;

    if (eventSet_==1) {
      // apply a miscalibration to all crystals and increment the 
      // ET sum, combined for all crystals
      for (int imiscal=0; imiscal<kNMiscalBinsEE; imiscal++) {
	if (miscalEE_[imiscal]*e> eCut_endc_ && et*miscalEE_[imiscal] < et_thr && goodCell_endc[hit.ix()-1][hit.iy()-1][sign]){
	  int ring = endcapRing_[hit.ix()-1][hit.iy()-1];
	  etsum_endc_miscal_[imiscal][ring] += miscalEE_[imiscal]*et;
	}
      }

      // spectra stuff
      if(spectra && hit.zside()>0) //POSITIVE!!!
	//      if(spectra && hit.zside()<0) //NEGATIVE!!!
	{
	  int ring = endcapRing_[hit.ix()-1][hit.iy()-1];

	  et_spectrum_e_histos[ring]->Fill(et*1000.);
	  e_spectrum_e_histos[ring]->Fill(e*1000.);
	  /*	  
	  if((hit.iy()-1)==50 && hit.ix()<50)
	    {
	      et_crystal_e_histos[ring]->Fill(et*1000.);
	      e_crystal_e_histos[ring]->Fill(e*1000.);
	    }
	  */
	  if(ring==16)
	    {
	      int iphi_endc = 0;
	      for (int ip=0; ip<nRing_[ring]; ip++) {
		if (phi==phi_endc[ip][ring]) iphi_endc=ip;
	      }
	      et_crystal_e_histos[iphi_endc]->Fill(et*1000.);
	      e_crystal_e_histos[iphi_endc]->Fill(e*1000.);
	    }
	}//if spectra

    }//if eventSet_==1
  }//for endc

}


//_____________________________________________________________________________

void PhiSymmetryCalibration::getKfactors()
{

  float epsilon_T_eb[kNMiscalBinsEB];
  float epsilon_M_eb[kNMiscalBinsEB];

  float epsilon_T_ee[kNMiscalBinsEE];
  float epsilon_M_ee[kNMiscalBinsEE];

  std::vector<TGraph*>  k_barl_graph(kBarlRings);
  std::vector<TCanvas*> k_barl_plot(kBarlRings);

  for (int ieta=0; ieta<kBarlRings; ieta++) {
    for (int imiscal=0; imiscal<kNMiscalBinsEB; imiscal++) {
      int middlebin =  int (kNMiscalBinsEB/2);
      epsilon_T_eb[imiscal] = etsum_barl_miscal_[imiscal][ieta]/etsum_barl_miscal_[middlebin][ieta] - 1.;
      epsilon_M_eb[imiscal] = miscalEB_[imiscal] - 1.;
    }
    k_barl_graph[ieta] = new TGraph (kNMiscalBinsEB,epsilon_M_eb,epsilon_T_eb);
    k_barl_graph[ieta]->Fit("pol1");


    ostringstream t;
    t<< "k_barl_" << ieta+1; 
    k_barl_plot[ieta] = new TCanvas(t.str().c_str(),"");
    k_barl_plot[ieta]->SetFillColor(10);
    k_barl_plot[ieta]->SetGrid();
    k_barl_graph[ieta]->SetMarkerSize(1.);
    k_barl_graph[ieta]->SetMarkerColor(4);
    k_barl_graph[ieta]->SetMarkerStyle(20);
    k_barl_graph[ieta]->GetXaxis()->SetLimits(-1.*kMiscalRangeEB,kMiscalRangeEB);
    k_barl_graph[ieta]->GetXaxis()->SetTitleSize(.05);
    k_barl_graph[ieta]->GetYaxis()->SetTitleSize(.05);
    k_barl_graph[ieta]->GetXaxis()->SetTitle("#epsilon_{M}");
    k_barl_graph[ieta]->GetYaxis()->SetTitle("#epsilon_{T}");
    k_barl_graph[ieta]->Draw("AP");

    k_barl_[ieta] = k_barl_graph[ieta]->GetFunction("pol1")->GetParameter(1);
    std::cout << "k_barl_[" << ieta << "]=" << k_barl_[ieta] << std::endl;
  }


  std::vector<TGraph*>  k_endc_graph(kEndcEtaRings);
  std::vector<TCanvas*> k_endc_plot(kEndcEtaRings);

  for (int ring=0; ring<kEndcEtaRings; ring++) {
    for (int imiscal=0; imiscal<kNMiscalBinsEE; imiscal++) {
      int middlebin =  int (kNMiscalBinsEE/2);
      epsilon_T_ee[imiscal] = etsum_endc_miscal_[imiscal][ring]/etsum_endc_miscal_[middlebin][ring] - 1.;
      epsilon_M_ee[imiscal] = miscalEE_[imiscal] - 1.;
    }
    k_endc_graph[ring] = new TGraph (kNMiscalBinsEE,epsilon_M_ee,epsilon_T_ee);
    k_endc_graph[ring]->Fit("pol1");

    ostringstream t;
    t<< "k_endc_"<< ring+1;
    k_endc_plot[ring] = new TCanvas(t.str().c_str(),"");
    k_endc_plot[ring]->SetFillColor(10);
    k_endc_plot[ring]->SetGrid();
    k_endc_graph[ring]->SetMarkerSize(1.);
    k_endc_graph[ring]->SetMarkerColor(4);
    k_endc_graph[ring]->SetMarkerStyle(20);
    k_endc_graph[ring]->GetXaxis()->SetLimits(-1*kMiscalRangeEE,kMiscalRangeEE);
    k_endc_graph[ring]->GetXaxis()->SetTitleSize(.05);
    k_endc_graph[ring]->GetYaxis()->SetTitleSize(.05);
    k_endc_graph[ring]->GetXaxis()->SetTitle("#epsilon_{M}");
    k_endc_graph[ring]->GetYaxis()->SetTitle("#epsilon_{T}");
    k_endc_graph[ring]->Draw("AP");

    k_endc_[ring] = k_endc_graph[ring]->GetFunction("pol1")->GetParameter(1);
    std::cout << "k_endc_[" << ring << "]=" << k_endc_[ring] << std::endl;
  }
 
  TFile f("PhiSymmetryCalibration_kFactors.root","recreate");
  for (int ieta=0; ieta<kBarlRings; ieta++) { 
    k_barl_plot[ieta]->Write();
    delete k_barl_plot[ieta]; 
    delete k_barl_graph[ieta];
  }
  for (int ring=0; ring<kEndcEtaRings; ring++) { 
    k_endc_plot[ring]->Write();
    delete k_endc_plot[ring];
    delete k_endc_graph[ring];
  }
  f.Close();

}


//_____________________________________________________________________________

void PhiSymmetryCalibration::fillHistos()
{
  TFile f("PhiSymmetryCalibration.root","recreate");

  std::vector<TH1F*> etsum_barl_histos(kBarlRings);
  std::vector<TH1F*> esum_barl_histos(kBarlRings);
  
  // determine ranges of ET sums to get histo bounds and book histos (barrel)
  for (int ieta=0; ieta<kBarlRings; ieta++) {
    float low=999999.;
    float high=0.;
    float low_e=999999.;
    float high_e=0.;

    for (int iphi=0; iphi<kBarlWedges; iphi++) {
      for (int sign=0; sign<kSides; sign++) {
	float etsum = etsum_barl_[ieta][iphi][sign];
	if (etsum<low && etsum!=0.) low=etsum;
	if (etsum>high) high=etsum;

	float esum = esum_barl_[ieta][iphi][sign];
	if (esum<low_e && esum!=0.) low_e=esum;
	if (esum>high_e) high_e=esum;
      }
    }
    
    ostringstream t;
    t << "etsum_barl_" << ieta+1;
    etsum_barl_histos[ieta]=new TH1F(t.str().c_str(),"",50,low-.2*low,high+.1*high);
    t.str("");

    t << "esum_barl_" << ieta+1;
    esum_barl_histos[ieta]=new TH1F(t.str().c_str(),"",50,low_e-.2*low_e,high_e+.1*high_e);
    t.str("");

    // fill barrel ET sum histos
    etsumMean_barl_[ieta]=0.;
    esumMean_barl_[ieta]=0.;
    for (int iphi=0; iphi<kBarlWedges; iphi++) {
      for (int sign=0; sign<kSides; sign++) {
	if(goodCell_barl[ieta][iphi][sign]){
	  float etsum = etsum_barl_[ieta][iphi][sign];
	  float esum  = esum_barl_[ieta][iphi][sign];
	  etsum_barl_histos[ieta]->Fill(etsum);
	  esum_barl_histos[ieta]->Fill(esum);
	  etsumMean_barl_[ieta]+=etsum;
	  esumMean_barl_[ieta]+=esum;
	}
      }
    }

    etsum_barl_histos[ieta]->Write();
    esum_barl_histos[ieta]->Write();
    etsumMean_barl_[ieta]/=(720.-nBads_barl[ieta]);
    esumMean_barl_[ieta]/=(720.-nBads_barl[ieta]);
    delete etsum_barl_histos[ieta];
    delete esum_barl_histos[ieta]; //VS
  }
  

  std::vector<TH1F*> etsum_endc_histos(kEndcEtaRings);
  std::vector<TH1F*> etsum_endc_uncorr_histos(kEndcEtaRings);
  std::vector<TH1F*> esum_endc_histos(kEndcEtaRings);

  std::vector<TH2F*> etsumvsarea_endc_histos(kEndcEtaRings);
  std::vector<TH2F*> esumvsarea_endc_histos(kEndcEtaRings);

  // determine ranges of ET sums to get histo bounds and book histos (endcap)
  for (int ring=0; ring<kEndcEtaRings; ring++) {

    float low=FLT_MAX;
    float low_uncorr=FLT_MAX;
    float high=0.;
    float high_uncorr=0;
    float low_e=FLT_MAX;
    float high_e=0.;
    float low_a=1.;
    float high_a=0.;
    for (int ix=0; ix<kEndcWedgesX; ix++) {
      for (int iy=0; iy<kEndcWedgesY; iy++) {
	if (endcapRing_[ix][iy]==ring) {
	  for (int sign=0; sign<kSides; sign++) {
	    float etsum = etsum_endc_[ix][iy][sign];
	    if (etsum<low && etsum!=0.) low=etsum;
	    if (etsum>high) high=etsum;

	    float etsum_uncorr = etsum_endc_uncorr[ix][iy][sign];
	    if (etsum_uncorr<low_uncorr && etsum_uncorr!=0.) low_uncorr=etsum_uncorr;
	    if (etsum_uncorr>high_uncorr) high_uncorr=etsum_uncorr;

	    float esum = esum_endc_[ix][iy][sign];
	    if (esum<low_e && esum!=0.) low_e=esum;
	    if (esum>high_e) high_e=esum;

	    float area = cellArea_[ix][iy];
	    if (area<low_a) low_a=area;
	    if (area>high_a) high_a=area;
	  }
	}
      }
    }

    ostringstream t;
    t<<"etsum_endc_" << ring+1;
    etsum_endc_histos[ring]= new TH1F(t.str().c_str(),"",50,low-.2*low,high+.1*high);
    t.str("");

    t<<"etsum_endc_uncorr_" << ring+1;
    etsum_endc_uncorr_histos[ring]= new TH1F(t.str().c_str(),"",50,low_uncorr-.2*low_uncorr,high_uncorr+.1*high_uncorr);
    t.str("");

    t<<"esum_endc_" << ring+1;
    esum_endc_histos[ring]= new TH1F(t.str().c_str(),"",50,low_e-.2*low_e,high_e+.1*high_e);
    t.str("");

    t<<"etsumvsarea_endc_" << ring+1;
    etsumvsarea_endc_histos[ring]= new TH2F(t.str().c_str(),";A_{#eta#phi};#Sigma E_{T}",50,low_a,high_a,50,low,high);
    t.str("");

    t<<"esumvsarea_endc_" << ring+1;
    esumvsarea_endc_histos[ring]= new TH2F(t.str().c_str(),";A_{#eta#phi};#Sigma E",50,low_a,high_a,50,low_e,high_e);
    t.str("");

    // fill endcap ET sum histos
    etsumMean_endc_[ring]=0.;
    esumMean_endc_[ring]=0.;
    for (int ix=0; ix<kEndcWedgesX; ix++) {
      for (int iy=0; iy<kEndcWedgesY; iy++) {
	if (endcapRing_[ix][iy]==ring) {
	  for (int sign=0; sign<kSides; sign++) {
	    if(goodCell_endc[ix][iy][sign]){
	      float etsum = etsum_endc_[ix][iy][sign];
	      float esum  = esum_endc_[ix][iy][sign];
	      float etsum_uncorr = etsum_endc_uncorr[ix][iy][sign];
	      etsum_endc_histos[ring]->Fill(etsum);
	      etsum_endc_uncorr_histos[ring]->Fill(etsum_uncorr);
	      esum_endc_histos[ring]->Fill(esum);
	      
	      float area = cellArea_[ix][iy];
	      etsumvsarea_endc_histos[ring]->Fill(area,etsum);
	      esumvsarea_endc_histos[ring]->Fill(area,esum);
	      
	      etsumMean_endc_[ring]+=etsum;
	      esumMean_endc_[ring]+=esum;
	    }
	  }
	}
      }
    }

    etsum_endc_histos[ring]->Write();
    etsum_endc_uncorr_histos[ring]->Write();
    esum_endc_histos[ring]->Write();
    etsumMean_endc_[ring]/=(float(nRing_[ring]*2-nBads_endc[ring]));
    esumMean_endc_[ring]/=(float(nRing_[ring]*2-nBads_endc[ring]));
    etsumvsarea_endc_histos[ring]->Write();
    esumvsarea_endc_histos[ring]->Write();

    delete etsum_endc_histos[ring];
    delete etsum_endc_uncorr_histos[ring];
    delete esum_endc_histos[ring];
    delete etsumvsarea_endc_histos[ring];
    delete esumvsarea_endc_histos[ring];
  }//ring


  // Maps of etsum in EB and EE
  TH2F barreletamap("barreletamap","barreletamap",171, -85,86,100,0,2);
  TH2F barrelmap("barrelmap","barrelmap - #frac{#Sigma E_{T}}{<#Sigma E_{T}>_{0}}",360,1,360, 171, -85,86);
  TH2F barrelmap_e("barrelmape","barrelmape - #frac{#Sigma E}{<#Sigma E>_{0}}",360,1,360, 171, -85,86);
  TH2F barrelmap_divided("barrelmapdiv","barrelmapdivided - #frac{#Sigma E_{T}}{hits}",360,1,360,171,-85,86);
  TH2F barrelmap_e_divided("barrelmapediv","barrelmapedivided - #frac{#Sigma E}{hits}",360,1,360,171,-85,86);
  TH2F endcmap_plus_corr("endcapmapplus_corrected","endcapmapplus - #frac{#Sigma E_{T}}{<#Sigma E_{T}>_{38}}",100,1,101,100,1,101);
  TH2F endcmap_minus_corr("endcapmapminus_corrected","endcapmapminus - #frac{#Sigma E_{T}}{<#Sigma E_{T}>_{38}}",100,1,101,100,1,101);
  TH2F endcmap_plus_uncorr("endcapmapplus_uncorrected","endcapmapplus_uncor - #frac{#Sigma E_{T}}{<#Sigma E_{T}>_{38}}",100,1,101,100,1,101);
  TH2F endcmap_minus_uncorr("endcapmapminus_uncorrected","endcapmapminus_uncor - #frac{#Sigma E_{T}}{<#Sigma E_{T}>_{38}}",100,1,101,100,1,101);
  TH2F endcmap_e_plus("endcapmapeplus","endcapmapeplus - #frac{#Sigma E}{<#Sigma E>_{38}}",100,1,101,100,1,101);
  TH2F endcmap_e_minus("endcapmapeminus","endcapmapeminus - #frac{#Sigma E}{<#Sigma E>_{38}}",100,1,101,100,1,101);

  for (int sign=0; sign<kSides; sign++) {

    int thesign = sign==1 ? 1:-1;

    for (int ieta=0; ieta<kBarlRings; ieta++) {
      for (int iphi=0; iphi<kBarlWedges; iphi++) {
	if(goodCell_barl[ieta][iphi][sign]){
	  barrelmap.Fill(iphi+1,ieta*thesign + thesign, etsum_barl_[ieta][iphi][sign]/etsumMean_barl_[0]);
	  barrelmap_e.Fill(iphi+1,ieta*thesign + thesign, esum_barl_[ieta][iphi][sign]/esumMean_barl_[0]); //VS
	  if (!nhits_barl_[ieta][iphi][sign]) nhits_barl_[ieta][iphi][sign] =1;
	  barrelmap_divided.Fill( iphi+1,ieta*thesign + thesign, etsum_barl_[ieta][iphi][sign]/nhits_barl_[ieta][iphi][sign]);
	  barrelmap_e_divided.Fill( iphi+1,ieta*thesign + thesign, esum_barl_[ieta][iphi][sign]/nhits_barl_[ieta][iphi][sign]); //VS
	  //int mod20= (iphi+1)%20;
	  //if (mod20==0 || mod20==1 ||mod20==2) continue;  // exclude SM boundaries
	  barreletamap.Fill(ieta*thesign + thesign,etsum_barl_[ieta][iphi][sign]/etsumMean_barl_[0]);
	}//if
      }//iphi
    }//ieta

     for (int ix=0; ix<kEndcWedgesX; ix++) {
      for (int iy=0; iy<kEndcWedgesY; iy++) {
	if (sign==1) {
	  endcmap_plus_corr.Fill(ix+1,iy+1,etsum_endc_[ix][iy][sign]/etsumMean_endc_[38]);
	  endcmap_plus_uncorr.Fill(ix+1,iy+1,etsum_endc_uncorr[ix][iy][sign]/etsumMean_endc_[38]);
	  endcmap_e_plus.Fill(ix+1,iy+1,esum_endc_[ix][iy][sign]/esumMean_endc_[38]);
	}
	else{ 
	  endcmap_minus_corr.Fill(ix+1,iy+1,etsum_endc_[ix][iy][sign]/etsumMean_endc_[38]);
	  endcmap_minus_uncorr.Fill(ix+1,iy+1,etsum_endc_uncorr[ix][iy][sign]/etsumMean_endc_[38]);
	  endcmap_e_minus.Fill(ix+1,iy+1,esum_endc_[ix][iy][sign]/esumMean_endc_[38]);
	}
      }//iy
     }//ix

  }  //sign
  
 

  barreletamap.Write();
  barrelmap_divided.Write();
  barrelmap.Write();
  barrelmap_e_divided.Write();
  barrelmap_e.Write();
  endcmap_plus_corr.Write();
  endcmap_minus_corr.Write();
  endcmap_plus_uncorr.Write();
  endcmap_minus_uncorr.Write();
  endcmap_e_plus.Write();
  endcmap_e_minus.Write();


  vector<TH1F*> etavsphi_endc(kEndcEtaRings);
  vector<TH1F*> areavsphi_endc(kEndcEtaRings);
  vector<TH1F*> etsumvsphi_endcp_corr(kEndcEtaRings);
  vector<TH1F*> etsumvsphi_endcm_corr(kEndcEtaRings);
  vector<TH1F*> etsumvsphi_endcp_uncorr(kEndcEtaRings);
  vector<TH1F*> etsumvsphi_endcm_uncorr(kEndcEtaRings);
  vector<TH1F*> esumvsphi_endcp(kEndcEtaRings);
  vector<TH1F*> esumvsphi_endcm(kEndcEtaRings);

  std::vector<TH1F*> deltaeta_histos(kEndcEtaRings);
  std::vector<TH1F*> deltaphi_histos(kEndcEtaRings);

  for(int ring =0; ring<kEndcEtaRings;++ring){
    
    ostringstream t;
    t<< "etavsphi_endc_"<<ring;
    etavsphi_endc[ring] = new TH1F(t.str().c_str(), t.str().c_str(),nRing_[ring],0,nRing_[ring]);
    t.str("");

    t<< "areavsphi_endc_"<<ring;
    areavsphi_endc[ring] = new TH1F(t.str().c_str(), t.str().c_str(),nRing_[ring],0,nRing_[ring]);
    t.str("");

    t<< "etsumvsphi_endcp_corr_"<<ring;
    etsumvsphi_endcp_corr[ring] = new TH1F(t.str().c_str(), t.str().c_str(),nRing_[ring],0,nRing_[ring]);
    t.str("");

    t << "etsumvsphi_endcm_corr_"<<ring;
    etsumvsphi_endcm_corr[ring] = new TH1F(t.str().c_str(), t.str().c_str(),nRing_[ring],0,nRing_[ring]);
    t.str("");

    t << "etsumvsphi_endcp_uncorr_"<<ring;
    etsumvsphi_endcp_uncorr[ring] = new TH1F(t.str().c_str(), t.str().c_str(),nRing_[ring],0,nRing_[ring]);
    t.str("");

    t << "etsumvsphi_endcm_uncorr_"<<ring;
    etsumvsphi_endcm_uncorr[ring] = new TH1F(t.str().c_str(), t.str().c_str(),nRing_[ring],0,nRing_[ring]);
    t.str("");
    
    t << "esumvsphi_endcp_"<<ring;
    esumvsphi_endcp[ring] = new TH1F(t.str().c_str(), t.str().c_str(),nRing_[ring],0,nRing_[ring]);
    t.str("");

    t << "esumvsphi_endcm_"<<ring;
    esumvsphi_endcm[ring] = new TH1F(t.str().c_str(), t.str().c_str(),nRing_[ring],0,nRing_[ring]);
    t.str("");

    t << "deltaeta_" << ring;
    deltaeta_histos[ring]= new TH1F(t.str().c_str(),"",50,-.1,.1);
    t.str("");
    t << "deltaphi_" << ring;
    deltaphi_histos[ring]= new TH1F(t.str().c_str(),"",50,-.1,.1);
    t.str("");
  }

  for (int ix=0; ix<kEndcWedgesX; ix++) {
      for (int iy=0; iy<kEndcWedgesY; iy++) {

	int ring = endcapRing_[ix][iy];
	if (ring!=-1) {
	  int iphi_endc=-1;
	  for (int ip=0; ip<nRing_[ring]; ip++) {
	    if (cellPhi_[ix][iy]==phi_endc[ip][ring]) iphi_endc=ip;
	  }

	  if(iphi_endc!=-1){
	    for (int sign=0; sign<kSides; sign++) {
	      if(goodCell_endc[ix][iy][sign]){
		if (sign==1){
		  etsumvsphi_endcp_corr[ring]->Fill(iphi_endc,etsum_endc_[ix][iy][sign]);
		  etsumvsphi_endcp_uncorr[ring]->Fill(iphi_endc,etsum_endc_uncorr[ix][iy][sign]);
		  esumvsphi_endcp[ring]->Fill(iphi_endc,esum_endc_[ix][iy][sign]);
		} else {
		  etsumvsphi_endcm_corr[ring]->Fill(iphi_endc,etsum_endc_[ix][iy][sign]);
		  etsumvsphi_endcm_uncorr[ring]->Fill(iphi_endc,etsum_endc_uncorr[ix][iy][sign]);
		  esumvsphi_endcm[ring]->Fill(iphi_endc,esum_endc_[ix][iy][sign]);
		}
	      }//if
	    }//sign
	    etavsphi_endc[ring]->Fill(iphi_endc,cellPos_[ix][iy].eta());
	    areavsphi_endc[ring]->Fill(iphi_endc,cellArea_[ix][iy]);
	  } //if iphi_endc

	  deltaeta_histos[ring]->Fill(delta_Eta[ix][iy]); //VS
	  deltaphi_histos[ring]->Fill(delta_Phi[ix][iy]); //VS
	}//if ring
      }//iy
  } //ix



 for(int ring =0; ring<kEndcEtaRings;++ring){
    
   etavsphi_endc[ring]->Write();
   areavsphi_endc[ring]->Write();
   etsumvsphi_endcp_corr[ring]->Write();
   etsumvsphi_endcm_corr[ring]->Write();
   etsumvsphi_endcp_uncorr[ring]->Write();
   etsumvsphi_endcm_uncorr[ring]->Write();
   esumvsphi_endcp[ring]->Write();
   esumvsphi_endcm[ring]->Write();
   deltaeta_histos[ring]->Write();
   deltaphi_histos[ring]->Write();

    
   delete etsumvsphi_endcp_corr[ring];
   delete etsumvsphi_endcm_corr[ring];
   delete etsumvsphi_endcp_uncorr[ring];
   delete etsumvsphi_endcm_uncorr[ring];
   delete etavsphi_endc[ring];
   delete areavsphi_endc[ring];
   delete esumvsphi_endcp[ring];
   delete esumvsphi_endcm[ring];
   delete deltaeta_histos[ring];
   delete deltaphi_histos[ring];
  }


  f.Close();
}


//_____________________________________________________________________________

void  PhiSymmetryCalibration::fillConstantsHistos(){
  
  TFile f("CalibHistos.root","recreate");  

  TH2F barreletamap("barreletamap","barreletamap",171, -85,86,100,0.,2.);
  TH2F barreletamapraw("barreletamapraw","barreletamapraw",171, -85,86,100,0.,2.);

  TH2F barrelmapold("barrelmapold","barrelmapold",360,1.,361.,171,-85.,86.);
  TH2F barrelmapnew("barrelmapnew","barrelmapnew",360,1.,361.,171,-85.,86.);
  TH2F barrelmapratio("barrelmapratio","barrelmapratio",360,1.,361.,171,-85.,86.);

  TH1F rawconst_endc_h("rawconst_endc","rawconst_endc",100,0.,2.);
  TH1F const_endc_h("const_endc","const_endc",100,0.,2.);

  TH1F oldconst_endc_h("oldconst_endc","oldconst_endc;oldCalib;",200,0,2);
  TH2F newvsraw_endc_h("newvsraw_endc","newvsraw_endc;rawConst;newCalib",200,0,2,200,0,2);

  TH2F endcapmapold_plus("endcapmapold_plus","endcapmapold_plus",100,1.,101.,100,1.,101.);
  TH2F endcapmapnew_plus("endcapmapnew_plus","endcapmapnew_plus",100,1.,101.,100,1.,101.);
  TH2F endcapmapratio_plus("endcapmapratio_plus","endcapmapratio_plus",100,1.,101.,100,1.,101.);

  TH2F endcapmapold_minus("endcapmapold_minus","endcapmapold_minus",100,1.,101.,100,1.,101.);
  TH2F endcapmapnew_minus("endcapmapnew_minus","endcapmapnew_minus",100,1.,101.,100,1.,101.);
  TH2F endcapmapratio_minus("endcapmapratio_minus","endcapmapratio_minus",100,1.,101.,100,1.,101.);


  for (int sign=0; sign<kSides; sign++) {

    int thesign = sign==1 ? 1:-1;

    for (int ieta=0; ieta<kBarlRings; ieta++) {
      for (int iphi=0; iphi<kBarlWedges; iphi++) {
	if(goodCell_barl[ieta][iphi][sign]){
	  //int mod20= (iphi+1)%20;
	  //if (mod20==0 || mod20==1 ||mod20==2) continue;  // exclude SM boundaries
	  barreletamap.Fill(ieta*thesign + thesign,newCalibs_barl[ieta][iphi][sign]);
	  barreletamapraw.Fill(ieta*thesign + thesign,rawconst_barl[ieta][iphi][sign]);
	  
	  barrelmapold.Fill(iphi+1,ieta*thesign + thesign, oldCalibs_barl[ieta][iphi][sign]);
	  barrelmapnew.Fill(iphi+1,ieta*thesign + thesign, newCalibs_barl[ieta][iphi][sign]);
	  barrelmapratio.Fill(iphi+1,ieta*thesign + thesign, newCalibs_barl[ieta][iphi][sign]/oldCalibs_barl[ieta][iphi][sign]);
	}//if
      }//iphi
    }//ieta

    for (int ix=0; ix<kEndcWedgesX; ix++) {
      for (int iy=0; iy<kEndcWedgesY; iy++) {
	if (goodCell_endc[ix][iy][sign]){
	  rawconst_endc_h.Fill(rawconst_endc[ix][iy][sign]);
	  const_endc_h.Fill(newCalibs_endc[ix][iy][sign]);
	  oldconst_endc_h.Fill(oldCalibs_endc[ix][iy][sign]);
	  newvsraw_endc_h.Fill(rawconst_endc[ix][iy][sign],newCalibs_endc[ix][iy][sign]);

	  if(sign==1){
	    endcapmapold_plus.Fill(ix+1,iy+1,oldCalibs_endc[ix][iy][sign]);
	    endcapmapnew_plus.Fill(ix+1,iy+1,newCalibs_endc[ix][iy][sign]);
	    endcapmapratio_plus.Fill(ix+1,iy+1,newCalibs_endc[ix][iy][sign]/oldCalibs_endc[ix][iy][sign]);
	  }
	  else{
	    endcapmapold_minus.Fill(ix+1,iy+1,oldCalibs_endc[ix][iy][sign]);
	    endcapmapnew_minus.Fill(ix+1,iy+1,newCalibs_endc[ix][iy][sign]);
	    endcapmapratio_minus.Fill(ix+1,iy+1,newCalibs_endc[ix][iy][sign]/oldCalibs_endc[ix][iy][sign]);
	  }

	}//if
      }//iy
    }//ix
    
  } // sides

  barreletamap.Write();
  barreletamapraw.Write();
  rawconst_endc_h.Write();
  const_endc_h.Write();
  oldconst_endc_h.Write();
  newvsraw_endc_h.Write();
  barrelmapold.Write();
  barrelmapnew.Write();
  barrelmapratio.Write();
  endcapmapold_plus.Write();
  endcapmapnew_plus.Write();
  endcapmapratio_plus.Write();
  endcapmapold_minus.Write();
  endcapmapnew_minus.Write();
  endcapmapratio_minus.Write();

  f.Close();
}

void PhiSymmetryCalibration::setUp(const edm::EventSetup& setup){


 // get initial constants out of DB
  EcalIntercalibConstantMap imap;
  if (eventSet_==0) {
    edm::ESHandle<EcalIntercalibConstants> pIcal;
    try {
      setup.get<EcalIntercalibConstantsRcd>().get(pIcal);
      std::cout << "Taken EcalIntercalibConstants" << std::endl;
      imap = pIcal.product()->getMap();
      std::cout << "imap.size() = " << imap.size() << std::endl;
    } catch ( std::exception& ex ) {     
      std::cerr << "Error! can't get EcalIntercalibConstants " << std::endl;
    }
  }


  // get the ecal geometry
  edm::ESHandle<CaloGeometry> geoHandle;
  setup.get<CaloGeometryRecord>().get(geoHandle);
  EcalRingCalibrationTools::setCaloGeometry(&(*geoHandle));   
  const CaloGeometry& geometry = *geoHandle;
  const CaloSubdetectorGeometry *barrelGeometry = geometry.getSubdetectorGeometry(DetId::Ecal, EcalBarrel);
  const CaloSubdetectorGeometry *endcapGeometry = geometry.getSubdetectorGeometry(DetId::Ecal, EcalEndcap);

  // channels status
  edm::ESHandle<EcalChannelStatus> chStatus;
  setup.get<EcalChannelStatusRcd>().get(chStatus);


  // loop over all barrel crystals
  barrelCells = geometry.getValidDetIds(DetId::Ecal, EcalBarrel);
  std::vector<DetId>::const_iterator barrelIt;
  for (barrelIt=barrelCells.begin(); barrelIt!=barrelCells.end(); barrelIt++) {
    EBDetId eb(*barrelIt);

    int sign = eb.zside()>0 ? 1 : 0;

    if (eventSet_==0) {
      // get the initial calibration constants
      EcalIntercalibConstantMap::const_iterator itcalib = imap.find(eb.rawId());
      if ( itcalib == imap.end() ) {
              // FIXME -- throw error
      }
      EcalIntercalibConstant calib = (*itcalib);
      oldCalibs_barl[abs(eb.ieta())-1][eb.iphi()-1][sign] = calib;
      if (eb.iphi()==1) std::cout << "Read old constant for crystal "
				  << " (" << eb.ieta() << "," << eb.iphi()
				  << ") : " << calib << std::endl;
    }

    // store eta value for each barrel ring
    if (eb.ieta()>0 &&eb.iphi()==1) {
      const CaloCellGeometry *cellGeometry = barrelGeometry->getGeometry(*barrelIt);
      cellEta_[eb.ieta()-1] = cellGeometry->getPosition().eta();
    }

    // store phi value for each barrel ring
    if (eb.ieta()==1 &&eb.iphi()>0) {
      const CaloCellGeometry *cellGeometry = barrelGeometry->getGeometry(*barrelIt);
      cellPhiB_[eb.iphi()-1] = cellGeometry->getPosition().phi(); //VS
    }

    // getting channel statuts information out of the DB (barrel)
    EcalChannelStatusMap::const_iterator chit = chStatus->find( *barrelIt );
    EcalChannelStatusCode chStatusCode = 10;
    if ( chit != chStatus->end() ) 
      {
	chStatusCode = *chit;
	int chs = (Int_t)(chStatusCode.getStatusCode() & 0x001F);
	if( chs <= 3 )
	  goodCell_barl[abs(eb.ieta())-1][eb.iphi()-1][sign] = true;
	//check
	//	std::cout << "ieta = "   << abs(eb.ieta()) << ", iphi = "     << eb.iphi()
	//		  << ", sign = " << sign           << ", chStatus = " << chs << std::endl;
      } 

    if( !goodCell_barl[abs(eb.ieta())-1][eb.iphi()-1][sign] )
      nBads_barl[abs(eb.ieta())-1]++;

  }

  // loop over all endcap crystals
  endcapCells = geometry.getValidDetIds(DetId::Ecal, EcalEndcap);
  std::vector<DetId>::const_iterator endcapIt;
  for (endcapIt=endcapCells.begin(); endcapIt!=endcapCells.end(); endcapIt++) {
    const CaloCellGeometry *cellGeometry = endcapGeometry->getGeometry(*endcapIt);
    EEDetId ee(*endcapIt);
    int ix=ee.ix()-1;
    int iy=ee.iy()-1;
    int sign = ee.zside()>0 ? 1 : 0;

    if (eventSet_==0) {
      // get the initial calibration constants
      EcalIntercalibConstantMap::const_iterator itcalib = imap.find(ee.rawId());
      if ( itcalib == imap.end() ) {
              // FIXME -- throw error
      }
      EcalIntercalibConstant calib = (*itcalib);
      oldCalibs_endc[ix][iy][sign] = calib;
      if (ix==49) std::cout << "Read old constant for xcrystal " //io metterei 50 (!)
			    << " (" << ix << "," << iy
			    << ") : " << calib << std::endl;
    }

    // store all crystal positions
    cellPos_[ix][iy] = cellGeometry->getPosition();
    cellPhi_[ix][iy] = cellGeometry->getPosition().phi();

    // calculate and store eta-phi area for each crystal front face Shoelace formuls
    const CaloCellGeometry::CornersVec& cellCorners (cellGeometry->getCorners()) ;
    cellArea_[ix][iy]=0.;
    delta_Eta[ix][iy]=0.;
    delta_Phi[ix][iy]=0.;
    for (int i=0; i<4; i++) {
      int iplus1 = i==3 ? 0 : i+1;
      cellArea_[ix][iy] += 
	cellCorners[i].eta()*float(cellCorners[iplus1].phi()) - 
	cellCorners[iplus1].eta()*float(cellCorners[i].phi());

      if(abs(cellCorners[i].eta()-cellCorners[iplus1].eta())>delta_Eta[ix][iy])
	delta_Eta[ix][iy] = cellCorners[i].eta()-cellCorners[iplus1].eta();
      if(abs(cellCorners[i].phi()-cellCorners[iplus1].phi())>delta_Phi[ix][iy])
	delta_Phi[ix][iy] = float(cellCorners[i].phi()-cellCorners[iplus1].phi());
    }
    cellArea_[ix][iy] = cellArea_[ix][iy]/2.;

    // getting channel statuts information out of the DB (endcap)
    EcalChannelStatusMap::const_iterator chit = chStatus->find( *endcapIt );
    EcalChannelStatusCode chStatusCode = 10;
    if ( chit != chStatus->end() ) 
      {
	chStatusCode = *chit;
	int chs = (Int_t)(chStatusCode.getStatusCode() & 0x001F);
	if( chs <= statusThreshold_ )
	  goodCell_endc[ix][iy][sign] = true;
	//check
	//	std::cout << "ix = "     << ix   << ", iy = "       << iy
	//		  << ", sign = " << sign << ", chStatus = " << chs << std::endl;
      } 

  }


  // get eta for each endcap ring
  //as defined in /CMSSW/Calibration/Tools/src/EcalRingCalibrationTools.cc
  float eta_ring[kEndcEtaRings];
  for (int ring=0; ring<kEndcEtaRings; ring++) {
    eta_ring[ring]=cellPos_[ring][50].eta();
  }

  
  // get eta boundaries for each endcap ring
  etaBoundary_[0]=1.479;
  etaBoundary_[39]=3.;  //It was 4. !!!
  for (int ring=1; ring<kEndcEtaRings; ring++) {
    etaBoundary_[ring]=(eta_ring[ring]+eta_ring[ring-1])/2.;
  }


  // determine to which ring each endcap crystal belongs,
  // the number of crystals in each ring,
  // and the mean eta-phi area of the crystals in each ring
  for (int ring=0; ring<kEndcEtaRings; ring++) {
    nRing_[ring]=0;
    meanCellArea_[ring]=0.;
    for (int ix=0; ix<kEndcWedgesX; ix++) {
      for (int iy=0; iy<kEndcWedgesY; iy++) {
        if (fabs(cellPos_[ix][iy].eta())>etaBoundary_[ring] &&
            fabs(cellPos_[ix][iy].eta())<etaBoundary_[ring+1]) {
          meanCellArea_[ring]+=cellArea_[ix][iy];
          endcapRing_[ix][iy]=ring;
          nRing_[ring]++;

	  for(int sign=0; sign<kSides; sign++){
	    if( !goodCell_endc[ix][iy][sign] )
	      nBads_endc[ring]++;
	  } //sign

        } //iy
      } //ix
    } //ring

    meanCellArea_[ring]/=nRing_[ring];

    std::cout << nRing_[ring] << " crystals with mean area " 
	      << meanCellArea_[ring] << " in endcap ring " << ring 
	      << " (" << etaBoundary_[ring] << "<eta<" << etaBoundary_[ring+1] << ")" << std::endl;
  }


  // fill phi_endc[ip][ring] vector
  for (int ring=0; ring<kEndcEtaRings; ring++) {
    
    for (int i=0; i<kMaxEndciPhi; i++) 
      phi_endc[i][ring]=0.;
    
    float philast=-999.;
    for (int ip=0; ip<nRing_[ring]; ip++) {
      float phimin=999.;
      for (int ix=0; ix<kEndcWedgesX; ix++) {
	for (int iy=0; iy<kEndcWedgesY; iy++) {
	  if (endcapRing_[ix][iy]==ring) {
	    if (cellPhi_[ix][iy]<phimin && cellPhi_[ix][iy]>philast) {
	      phimin=cellPhi_[ix][iy];
	    } //if edges
	  } //if ring
	} //iy
      } //ix	
      phi_endc[ip][ring]=phimin;
      philast=phimin;
    } //ip
  
  } //ring


  // print endcap geometry
  if (eventSet_==0){
    ofstream endcapgeom("endcapgeom.dat",ios::out);
    for (int ix=0; ix<kEndcWedgesX; ix++) {
      for (int iy=0; iy<kEndcWedgesY; iy++) {
	
	endcapgeom << " ix " <<ix << " iy "<< iy  
		   << " area " << cellArea_[ix][iy]<< " ring "
		   << endcapRing_[ix][iy]<< " phi "
		   << cellPhi_[ix][iy] << endl;  
      } //ix
    } //iy
  } // if


  /*  
  // Stefano rough reiteration attempt
  if (reiteration_){
    edm::ESHandle<EcalIntercalibConstants> pIcal;
    iSetup.get<EcalIntercalibConstantsRcd>().get(pIcal);
    previousCalibs_  = *(pIcal.product());
  }
  */
}
