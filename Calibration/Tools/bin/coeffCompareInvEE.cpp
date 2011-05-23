#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"
#include "Calibration/Tools/interface/calibXMLwriter.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/CaloMiscalibTools.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/CaloMiscalibMapEcal.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/MiscalibReaderFromXMLEcalBarrel.h"
#include "CalibCalorimetry/CaloMiscalibTools/interface/MiscalibReaderFromXMLEcalEndcap.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

//#include "Calibration/EcalAlCaRecoProducers/interface/trivialParser.h"
//#include "Calibration/EcalAlCaRecoProducers/bin/trivialParser.h"

#include "TH2.h"
#include "TH1.h"
#include "TFile.h"
#include "TProfile.h"

#define PI_GRECO 3.14159265

inline int etaShifter (const int etaOld) 
   {
     if (etaOld < 0) return etaOld + 85 ;
     else if (etaOld > 0) return etaOld + 84 ;
     assert(false);
   }


// ------------------------------------------------------------------------

// MF questo file prende due set di coefficienti e li confronta 
// 
 
//-------------------------------------------------------------------------

int main (int argc, char* argv[]) 
{

  int EEradStart = 15 ;
  int EEradEnd = 50 ;
  int EEphiStart = 20 ;
  int EEphiEnd = 45 ;

  std::string endcapfile = argv[1] ; 
  std::string calibEndcapfile = argv[2] ; 

//  std::string endcapfile = "/afs/cern.ch/user/g/govoni/scratch1/CMSSW/CALIB/CMSSW_1_6_0/src/CalibCalorimetry/CaloMiscalibTools/data/ecal_endcap_startup.xml" ;
//  std::string calibEndcapfile = "/afs/cern.ch/user/g/govoni/scratch1/CMSSW/CALIB/CMSSW_1_6_0/src/CalibCalorimetry/CaloMiscalibTools/data/inv_ecal_endcap_startup.xml" ;
  
  //PG get the miscalibration files for EB and EE
  //PG ------------------------------------------

  CaloMiscalibMapEcal EEscalibMap ;
  EEscalibMap.prefillMap () ;
  MiscalibReaderFromXMLEcalEndcap endcapreader (EEscalibMap) ;
  if (!endcapfile.empty ()) endcapreader.parseXMLMiscalibFile (endcapfile) ;
  EcalIntercalibConstants* EEconstants = 
         new EcalIntercalibConstants (EEscalibMap.get ()) ;
  EcalIntercalibConstantMap iEEscalibMap = EEconstants->getMap () ;  //MF prende i vecchi coeff

  //PG get the recalibration files for EB and EE
  //PG -----------------------------------------

  CaloMiscalibMapEcal EEcalibMap ;
  EEcalibMap.prefillMap () ;
  MiscalibReaderFromXMLEcalEndcap calibEndcapreader (EEcalibMap) ;
  if (!calibEndcapfile.empty ()) calibEndcapreader.parseXMLMiscalibFile (calibEndcapfile) ;
  EcalIntercalibConstants* EECconstants = 
         new EcalIntercalibConstants (EEcalibMap.get ()) ;
  EcalIntercalibConstantMap iEEcalibMap = EECconstants->getMap () ;  //MF prende i vecchi coeff
  
  //PG fill the histograms
  //PG -------------------
  
  TH1F EEPCompareCoeffDistr ("EEPCompareCoeffDistr","EEPCompareCoeffDistr",5000,0,2) ;
  TH2F EEPCompareCoeffMap ("EEPCompareCoeffMap","EEPCompareCoeffMap",101,0,101,101,0,101) ;
  TH2F EEPCompareCoeffEtaTrend ("EEPCompareCoeffEtaTrend",
                               "EEPCompareCoeffEtaTrend",
                               51,0,50,500,0,2) ;
  TProfile EEPCompareCoeffEtaProfile ("EEPCompareCoeffEtaProfile",
                                      "EEPCompareCoeffEtaProfile",
                                      51,0,50,0,2) ;
   
  // ECAL endcap +
  for (int ix = 1 ; ix <= 100 ; ++ix)
   for (int iy = 1 ; iy <= 100 ; ++iy)
    {
      int rad = static_cast<int> (sqrt ((ix - 50) * (ix - 50) +
                                        (iy - 50) * (iy - 50))) ;
      if (rad < EEradStart || rad > EEradEnd) continue ;
      double phiTemp = atan2 (iy - 50, ix - 50) ;
      if (phiTemp < 0) phiTemp += 2 * PI_GRECO ;
      int phi = static_cast<int> ( phiTemp * 180 / PI_GRECO) ;
      if (phi < EEphiStart || phi > EEphiEnd) continue ;
      if (!EEDetId::validDetId (ix,iy,1)) continue ;
      EEDetId det = EEDetId (ix, iy, 1, EEDetId::XYMODE) ;
      if (*(iEEscalibMap.find (det.rawId ())) == 0) continue ;
      double factor = *(iEEcalibMap.find (det.rawId ()))/ 
                      *(iEEscalibMap.find (det.rawId ())) ;
      EEPCompareCoeffDistr.Fill (factor) ;
      EEPCompareCoeffMap.Fill (ix,iy,factor) ;
      EEPCompareCoeffEtaTrend.Fill (rad,factor) ;
      EEPCompareCoeffEtaProfile.Fill (rad,factor) ;
    } // ECAL endcap +

  // ECAL endcap-
  TH1F EEMCompareCoeffDistr ("EEMCompareCoeffDistr","EEMCompareCoeffDistr",200,0,2) ;
  TH2F EEMCompareCoeffMap ("EEMCompareCoeffMap","EEMCompareCoeffMap",100,0,100,100,0,100) ;
  TH2F EEMCompareCoeffEtaTrend ("EEMCompareCoeffEtaTrend",
                               "EEMCompareCoeffEtaTrend",
                               51,0,50,500,0,2) ;
  TProfile EEMCompareCoeffEtaProfile ("EEMCompareCoeffEtaProfile",
                                     "EEMCompareCoeffEtaProfile",
                                     51,0,50,0,2) ;
   
  // ECAL endcap -
  for (int ix = 1 ; ix <= 100 ; ++ix)
   for (int iy = 1 ; iy <= 100 ; ++iy)
    {
      int rad = static_cast<int> (sqrt ((ix - 50) * (ix - 50) +
                                        (iy - 50) * (iy - 50))) ;
      if (rad < EEradStart || rad > EEradEnd) continue ;
      double phiTemp = atan2 (iy - 50, ix - 50) ;
      if (phiTemp < 0) phiTemp += 2 * PI_GRECO ;
      if (!EEDetId::validDetId (ix,iy,-1)) continue ;
      EEDetId det = EEDetId (ix, iy, -1, EEDetId::XYMODE) ;
      if (*(iEEscalibMap.find (det.rawId ())) == 0) continue ;
      double factor = *(iEEcalibMap.find (det.rawId ())) / 
                      *(iEEscalibMap.find (det.rawId ())) ;
      EEMCompareCoeffDistr.Fill (factor) ;
      EEMCompareCoeffMap.Fill (ix,iy,factor) ;
      EEMCompareCoeffEtaTrend.Fill (rad,factor) ;
      EEMCompareCoeffEtaProfile.Fill (rad,factor) ;
    } // ECAL endcap -
    
 std::string filename = "coeffcompareEE.root" ;
 TFile out (filename.c_str (),"recreate") ;
 EEMCompareCoeffMap.Write() ;
 EEMCompareCoeffDistr.Write() ;
 EEMCompareCoeffEtaTrend.Write () ;
 EEMCompareCoeffEtaProfile.Write () ;
 EEPCompareCoeffMap.Write() ;
 EEPCompareCoeffDistr.Write() ;
 EEPCompareCoeffEtaTrend.Write () ;
 EEPCompareCoeffEtaProfile.Write () ;
 out.Close() ;
  
}
