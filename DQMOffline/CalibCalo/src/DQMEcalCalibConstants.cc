/** Implementation of class  DQMEcalCalibConstants 
 *
 *  \author Stefano Argiro
 *  \date   $Date$
 *  \version $Id$ 
 */ 
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ServiceRegistry/interface/Service.h" 
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "DQMOffline/CalibCalo/src/DQMEcalCalibConstants.h"

using namespace std;
using namespace edm;

DQMEcalCalibConstants::DQMEcalCalibConstants(const edm::ParameterSet& ps):

  folderName_(ps.getUntrackedParameter<string>("FolderName","AlCa/EcalCalib")),
  saveToFile_(ps.getUntrackedParameter<bool>("SaveToFile",false)),
  fileName_  (ps.getUntrackedParameter<string>("FileName","EcalCalib.root")),
  DBlabel_   (ps.getUntrackedParameter<string>("DBLabel","EcalCalib.root")),
  RefDBlabel_(ps.getUntrackedParameter<string>("RefDBLabel","EcalCalib.root"))
  {
    dbe_ = Service<DQMStore>().operator->();
  }


void DQMEcalCalibConstants::beginJob(const edm::EventSetup& c){

   cDistrEB_ = dbe_->book1D("coeffDistrEB","coeffDistrEB",500,0,2);
   cDistrEE_ = dbe_->book1D("coeffDistrEE","coeffDistrEE",500,0,2) ;
   cMapEB_   = dbe_->book2D("coeffMapEB","coeffMapEB",171,-85,86,360,1,361);
   cMapEE_   = dbe_->book2D("coeffMapEE","coeffMapEE",101,0,101,101,0,101);  


   compCDistrEB_ = dbe_->book1D("Ebcomparecoeff","EBCompareCoeff",1000,0,2);   
   compCMapEB_   = dbe_->book2D("EBCompareCoeffMap",
				"EBCompareCoeffMap",
				360,1,361,171,-85,86);
   compCEtaTrendEB_= dbe_->book2D("EBCompareCoeffEtaTrend",
				  "EBCompareCoeffEtaTrend",
				  171,-85,86,500,0,2);
   compCEtaProfileEB_ = dbe_->bookProfile("EBCompareCoeffEtaProfile",
					  "EBCompareCoeffEtaProfile",
					  171,-85,86,0,2);
   compCDistrM1_ =dbe_->book1D("EBCompareCoeff_M1","EBCompareCoeff_M1",
			       1000,0,2) ;
   compCDistrM2_ =dbe_->book1D("EBCompareCoeff_M2","EBCompareCoeff_M2",
			       1000,0,2) ;
   compCDistrM3_ =dbe_->book1D("EBCompareCoeff_M3","EBCompareCoeff_M3",
			       1000,0,2) ;
   compCDistrM4_ =dbe_->book1D("EBCompareCoeff_M4","EBCompareCoeff_M4",
			       1000,0,2) ;


   compCDistrEEP_ = dbe_->book1D("EEPCompareCoeffDistr",
				 "EEPCompareCoeffDistr",5000,0,2) ;
   compCMapEEP_ = dbe_->book2D("EEPCompareCoeffMap",
			       "EEPCompareCoeffMap",101,0,101,101,0,101);
   compCEtaTrendEEP_ =dbe_->book2D("EEPCompareCoeffEtaTrend",
				   "EEPCompareCoeffEtaTrend",
				   51,0,50,500,0,2) ;
   compCEtaProfileEEP_ = dbe_->bookProfile("EEPCompareCoeffEtaProfile",
					   "EEPCompareCoeffEtaProfile",
					   51,0,50,0,2) ;;
   compCDistrR1P_ = dbe_->book1D("EEPCompareCoeff_R1",
				 "EEPCompareCoeff_R1",1000,0,2);
   compCDistrR2P_ = dbe_->book1D("EEPCompareCoeff_R1",
				 "EEPCompareCoeff_R2",1000,0,2);
   compCDistrR3P_ = dbe_->book1D("EEPCompareCoeff_R1",
				 "EEPCompareCoeff_R3",1000,0,2);
   compCDistrR4P_ = dbe_->book1D("EEPCompareCoeff_R1",
				 "EEPCompareCoeff_R4",1000,0,2);
   compCDistrR5P_ = dbe_->book1D("EEPCompareCoeff_R1",
				 "EEPCompareCoeff_R5",1000,0,2);
   
   
   compCDistrEEM_ = dbe_->book1D("EEMCompareCoeffDistr",
				 "EEMCompareCoeffDistr",5000,0,2) ;
   compCMapEEM_ = dbe_->book2D("EEMCompareCoeffMap",
			       "EEMCompareCoeffMap",101,0,101,101,0,101);
   compCEtaTrendEEM_ =dbe_->book2D("EEMCompareCoeffEtaTrend",
				   "EEMCompareCoeffEtaTrend",
				   51,0,50,500,0,2) ;
   compCEtaProfileEEM_ = dbe_->bookProfile("EEMCompareCoeffEtaProfile",
					   "EEMCompareCoeffEtaProfile",
					   51,0,50,0,2) ;;
   compCDistrR1M_ = dbe_->book1D("EEMCompareCoeff_R1",
				 "EEMCompareCoeff_R1",1000,0,2);
   compCDistrR2M_ = dbe_->book1D("EEMCompareCoeff_R1",
				 "EEMCompareCoeff_R2",1000,0,2);
   compCDistrR3M_ = dbe_->book1D("EEMCompareCoeff_R1",
				 "EEMCompareCoeff_R3",1000,0,2);
   compCDistrR4M_ = dbe_->book1D("EEMCompareCoeff_R1",
				 "EEMCompareCoeff_R4",1000,0,2);
   compCDistrR5M_ = dbe_->book1D("EEMCompareCoeff_R1",
				 "EEMCompareCoeff_R5",1000,0,2);

}

void DQMEcalCalibConstants::beginRun(const edm::Run& r, const edm::EventSetup& c){
   
  const double PI = acos(-1);

  // set of constants
  edm::ESHandle<EcalIntercalibConstants> intercalib;
  c.get<EcalIntercalibConstantsRcd>().get(DBlabel_,intercalib);
  EcalIntercalibConstantMap cmap = intercalib->getMap () ;  

  // reference set of constants
  edm::ESHandle<EcalIntercalibConstants> ref_intercalib;
  c.get<EcalIntercalibConstantsRcd>().get(RefDBlabel_,ref_intercalib);
  EcalIntercalibConstantMap ref_cmap = ref_intercalib->getMap () ;  
  

  // 100% recycled code


  int EBetaStart = 1 ;
  int EBetaEnd = 86 ;
  int EBphiStart = 0 ;
  int EBphiEnd = 360 ;
  int power = -1 ;

  int EEradStart = 15 ;
  int EEradEnd = 50 ;
  int EEphiStart = 0 ;
  int EEphiEnd = 360 ;

  

  // plot EB coefficients
  for (int ieta =- 85 ; ieta <= 85 ; ++ieta)
    for (int iphi = 1 ; iphi <= 360 ; ++iphi)
      {      
	if (!EBDetId::validDetId (ieta,iphi)) continue ;
	EBDetId det = EBDetId (ieta,iphi,EBDetId::ETAPHIMODE) ;
	double coeff = (*(cmap.find (det.rawId ()))) ;
	cDistrEB_->Fill (coeff) ;
	cMapEB_->Fill (ieta,iphi,coeff) ;
      } 
  
  // plot EE coefficients
  for (int ix = 1 ; ix <= 100 ; ++ix)
    for (int iy = 1 ; iy <= 100 ; ++iy)
      {      
	int rad = int (sqrt ((ix - 50) * (ix - 50) +
			     (iy - 50) * (iy - 50))) ;
	if (rad < EEradStart || rad > EEradEnd) continue ;
	double phiTemp = atan2 (iy - 50, ix - 50) ;
	if (phiTemp < 0) phiTemp += 2 * PI ;
	int phi = int( phiTemp * 180 / PI) ;
	if (phi < EEphiStart || phi > EEphiEnd) continue ;
	
	if (!EEDetId::validDetId (ix,iy,1)) continue ;
	EEDetId det = EEDetId (ix,iy,1,EEDetId::XYMODE) ;
	double coeff = (*(cmap.find (det.rawId ())));

	cDistrEE_->Fill (coeff) ;
	cMapEE_->Fill (ix,iy,coeff) ;
      } 

  
  //  compare EB

 
  

  for (int ieta = EBetaStart ; ieta < EBetaEnd ; ++ieta)
    {
      double phiSum = 0. ; 
      double phiSumSq = 0. ; 
      double N = 0. ;
      for (int iphi = EBphiStart ; iphi <= EBphiEnd ; ++iphi)
	{
          if (!EBDetId::validDetId (ieta,iphi)) continue ;
          EBDetId det = EBDetId (ieta,iphi,EBDetId::ETAPHIMODE) ;
          double factor = *(cmap.find (det.rawId ())) * 
	    *(ref_cmap.find (det.rawId ())) ;
          if (power != 1 && factor != 0) 
	    factor = *(cmap.find (det.rawId ())) / 
	      *(ref_cmap.find (det.rawId ()));
          
          phiSum += factor ;
          phiSumSq += factor * factor ;
          N += 1. ;
          compCDistrEB_->Fill (factor) ;
	  compCMapEB_->Fill (iphi,ieta,factor) ;
	  compCEtaTrendEB_->Fill (ieta,factor) ;
          compCEtaProfileEB_->Fill (ieta,factor) ;

          if (abs(ieta) < 26) compCDistrM1_->Fill (factor) ;
          else if (abs(ieta) < 46) compCDistrM2_->Fill (factor) ;
          else if (abs(ieta) < 66) compCDistrM3_->Fill (factor) ;
          else compCDistrM4_->Fill (factor) ;
	} //loop over phi
  } //loop over eta

 
  // compare EE
  // ECAL endcap +
  for (int ix = 1 ; ix <= 100 ; ++ix)
      for (int iy = 1 ; iy <= 100 ; ++iy)
  {
      int rad = int (sqrt ((ix - 50) * (ix - 50) +
              (iy - 50) * (iy - 50))) ;
      if (rad < EEradStart || rad > EEradEnd) continue ;
      double phiTemp = atan2 (iy - 50, ix - 50) ;
      if (phiTemp < 0) phiTemp += 2 * PI ;
      int phi = int ( phiTemp * 180 / PI) ;
      if (phi < EEphiStart || phi > EEphiEnd) continue ;
      if (!EEDetId::validDetId (ix,iy,1)) continue ;
      EEDetId det = EEDetId (ix, iy, 1, EEDetId::XYMODE) ;
      double factor = *(cmap.find (det.rawId ())) / 
                  *(ref_cmap.find (det.rawId ())) ;
      
      compCDistrEEP_->Fill (factor) ;
      compCMapEEP_->Fill (ix,iy,factor) ;
      compCEtaTrendEEP_->Fill (rad,factor) ;
      compCEtaProfileEEP_->Fill (rad,factor) ;
      if (abs(rad) < 22) continue ;
      else if (abs(rad) < 27) compCDistrR1P_->Fill (factor) ;
      else if (abs(rad) < 32) compCDistrR2P_->Fill (factor) ;
      else if (abs(rad) < 37) compCDistrR3P_->Fill (factor) ;
      else if (abs(rad) < 42) compCDistrR4P_->Fill (factor) ;
      else compCDistrR5P_->Fill (factor) ;

  } // ECAL endcap +

 
   
  // compare EE
  // ECAL endcap -
  for (int ix = 1 ; ix <= 100 ; ++ix)
      for (int iy = 1 ; iy <= 100 ; ++iy)
  {
      int rad = int (sqrt ((ix - 50) * (ix - 50) +
              (iy - 50) * (iy - 50))) ;
      if (rad < EEradStart || rad > EEradEnd) continue ;
      double phiTemp = atan2 (iy - 50, ix - 50) ;
      if (phiTemp < 0) phiTemp += 2 * PI ;
      int phi = int ( phiTemp * 180 / PI) ;
      if (phi < EEphiStart || phi > EEphiEnd) continue ;
      if (!EEDetId::validDetId (ix,iy,1)) continue ;
      EEDetId det = EEDetId (ix, iy, -1, EEDetId::XYMODE) ;
      double factor = *(cmap.find (det.rawId ())) / 
                  *(ref_cmap.find (det.rawId ())) ;
      
      compCDistrEEM_->Fill (factor) ;
      compCMapEEM_->Fill (ix,iy,factor) ;
      compCEtaTrendEEM_->Fill (rad,factor) ;
      compCEtaProfileEEM_->Fill (rad,factor) ;
      if (abs(rad) < 22) continue ;
      else if (abs(rad) < 27) compCDistrR1M_->Fill (factor) ;
      else if (abs(rad) < 32) compCDistrR2M_->Fill (factor) ;
      else if (abs(rad) < 37) compCDistrR3M_->Fill (factor) ;
      else if (abs(rad) < 42) compCDistrR4M_->Fill (factor) ;
      else compCDistrR5M_->Fill (factor) ;

  } // ECAL endcap -


  

}



void DQMEcalCalibConstants::endJob(){
  
  if (saveToFile_) {
     dbe_->save(fileName_);
  }
  
}

DQMEcalCalibConstants::~DQMEcalCalibConstants(){}
