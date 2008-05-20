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
#include "TProfile.h"
#include "TH1.h"
#include "TFile.h"

#define PI_GRECO 3.14159265

inline int etaShifter (const int etaOld) 
   {
     if (etaOld < 0) return etaOld + 85 ;
     else if (etaOld > 0) return etaOld + 84 ;
   }


// ------------------------------------------------------------------------

// MF questo file prende due set di coefficienti e li confronta 
// 
 
//-------------------------------------------------------------------------

int main (int argc, char* argv[]) 
{
  int EBetaStart = 1 ;
  int EBetaEnd = 86 ;
  int EBphiStart = 20 ;
  int EBphiEnd = 60 ;
  int power = 1 ;

  std::string barrelfile = argv[1] ; 
  std::string calibBarrelfile = argv[2] ; 
  std::string filename = "coeffcompareEB.root" ;

  if (argc > 3)
    {
      if (argc < 7)
        {
          std::cerr << "Too few (or too many) arguments passed" << std::endl ;
          exit (1) ;
        }
      EBetaStart = atoi (argv[3]) ;
      EBetaEnd = atoi (argv[4]) ;
      EBphiStart = atoi (argv[5]) ;
      EBphiEnd = atoi (argv[6]) ;
    }

  if (argc == 8) power = atoi (argv[7]) ;
  if (argc == 9) filename = argv[8] ;

//  std::string barrelfile = "/afs/cern.ch/user/g/govoni/scratch1/CMSSW/CALIB/CMSSW_1_6_0/src/CalibCalorimetry/CaloMiscalibTools/data/ecal_barrel_startup.xml" ;
//  std::string calibBarrelfile = "/afs/cern.ch/user/g/govoni/scratch1/CMSSW/CALIB/CMSSW_1_6_0/src/CalibCalorimetry/CaloMiscalibTools/data/inv_ecal_barrel_startup.xml" ;
  
  //PG get the miscalibration files for EB and EE
  //PG ------------------------------------------

  CaloMiscalibMapEcal EBscalibMap ;
  EBscalibMap.prefillMap () ;
  MiscalibReaderFromXMLEcalBarrel barrelreader (EBscalibMap) ;
  if (!barrelfile.empty ()) barrelreader.parseXMLMiscalibFile (barrelfile) ;
  EcalIntercalibConstants* EBconstants = 
         new EcalIntercalibConstants (EBscalibMap.get ()) ;
  EcalIntercalibConstantMap iEBscalibMap = EBconstants->getMap () ;  //MF prende i vecchi coeff

  //PG get the recalibration files for EB and EE
  //PG -----------------------------------------

  CaloMiscalibMapEcal EBcalibMap ;
  EBcalibMap.prefillMap () ;
  MiscalibReaderFromXMLEcalBarrel calibBarrelreader (EBcalibMap) ;
  if (!calibBarrelfile.empty ()) calibBarrelreader.parseXMLMiscalibFile (calibBarrelfile) ;
  EcalIntercalibConstants* EBCconstants = 
         new EcalIntercalibConstants (EBcalibMap.get ()) ;
  EcalIntercalibConstantMap iEBcalibMap = EBCconstants->getMap () ;  //MF prende i vecchi coeff

  //PG fill the histograms
  //PG -------------------
  
  // ECAL barrel
  TH1F EBCompareCoeffDistr ("EBCompareCoeff","EBCompareCoeff",1000,0,2) ;
  TH2F EBCompareCoeffMap ("EBCompareCoeffMap","EBCompareCoeffMap",171,-85,86,360,1,361) ;
  TH2F EBCompareCoeffEtaTrend ("EBCompareCoeffEtaTrend","EBCompareCoeffEtaTrend",
                               171,-85,86,500,0,2) ;
  TProfile EBCompareCoeffEtaProfile ("EBCompareCoeffEtaProfile","EBCompareCoeffEtaProfile",
                                     171,-85,86,0,2) ;
  TH1F EBCompareCoeffDistr_M1 ("EBCompareCoeff_M1","EBCompareCoeff_M1",1000,0,2) ;
  TH1F EBCompareCoeffDistr_M2 ("EBCompareCoeff_M2","EBCompareCoeff_M2",1000,0,2) ;
  TH1F EBCompareCoeffDistr_M3 ("EBCompareCoeff_M3","EBCompareCoeff_M3",1000,0,2) ;
  TH1F EBCompareCoeffDistr_M4 ("EBCompareCoeff_M4","EBCompareCoeff_M4",1000,0,2) ;
  
  // ECAL barrel
  
  //PG loop over eta
  for (int ieta = EBetaStart ; ieta < EBetaEnd ; ++ieta)
    {
      double phiSum = 0. ; 
      double phiSumSq = 0. ; 
      double N = 0. ;
      for (int iphi = EBphiStart ; iphi <= EBphiEnd ; ++iphi)
        {
          if (!EBDetId::validDetId (ieta,iphi)) continue ;
          EBDetId det = EBDetId (ieta,iphi,EBDetId::ETAPHIMODE) ;
          double factor = *(iEBcalibMap.find (det.rawId ())) * 
                          *(iEBscalibMap.find (det.rawId ())) ;
          if (power != 1 && factor != 0) 
            factor = *(iEBcalibMap.find (det.rawId ())) / 
                     *(iEBscalibMap.find (det.rawId ()));
          phiSum += factor ;
          phiSumSq += factor * factor ;
          N += 1. ;
          EBCompareCoeffDistr.Fill (factor) ;
          EBCompareCoeffMap.Fill (ieta,iphi,factor) ;
          EBCompareCoeffEtaTrend.Fill (ieta,factor) ;
          EBCompareCoeffEtaProfile.Fill (ieta,factor) ;
          if (abs(ieta) < 26) EBCompareCoeffDistr_M1.Fill (factor) ;
          else if (abs(ieta) < 46) EBCompareCoeffDistr_M2.Fill (factor) ;
          else if (abs(ieta) < 66) EBCompareCoeffDistr_M3.Fill (factor) ;
          else EBCompareCoeffDistr_M4.Fill (factor) ;
        } //PG loop over phi
       double phiMean = phiSum / N ;
       double phiRMS = sqrt (phiSumSq / N - phiMean * phiMean) ;        
    } //PG loop over eta

  // trend vs eta FIXME to be renormalized
  TH1D * EBEtaTrend = EBCompareCoeffMap.ProjectionX () ;
  // trend vs phi FIXME to be renormalized
  TH1D * EBPhiTrend = EBCompareCoeffMap.ProjectionY () ;

  TFile out (filename.c_str (),"recreate") ;
  EBCompareCoeffMap.Write () ;
  EBCompareCoeffDistr.Write () ;  
  EBCompareCoeffEtaTrend.Write () ;
  EBCompareCoeffEtaProfile.Write () ;
  EBCompareCoeffDistr_M1.Write () ;
  EBCompareCoeffDistr_M2.Write () ;
  EBCompareCoeffDistr_M3.Write () ;
  EBCompareCoeffDistr_M4.Write () ;
  EBEtaTrend->Write () ;
  EBPhiTrend->Write () ;
  out.Close () ;
  
}
