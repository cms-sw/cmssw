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




//---- new for IOV Iterator ----
#include "CondCore/Utilities/interface/CondIter.h"
#include "TSystem.h"




#include "TH2.h"
#include "TProfile.h"
#include "TH1.h"
#include "TFile.h"

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
    
    int EBetaStart = 1 ;
    int EBetaEnd = 86 ;
    int EBphiStart = 0 ;
    int EBphiEnd = 360 ;
    int power = -1 ;
    
    
    std::string filename = "coeffcompareEB.root" ;
   
//     std::string NameDBOracle1 = "oracle://cms_orcoff_int2r/CMS_COND_ECAL";
//     std::string TagDBOracle1 = "EcalIntercalibConstants_startup_csa08_mc";  
    
//     std::string NameDBOracle2 = "oracle://cms_orcoff_int2r/CMS_COND_ECAL";
//     std::string TagDBOracle2 = "EcalIntercalibConstants_inv_startup_csa08_mc";  
 
    
    std::string NameDBOracle1 = "oracle://cms_orcoff_prod/CMS_COND_20X_ECAL";
    std::string TagDBOracle1 = "EcalIntercalibConstants_phi_Zee_csa08_s156_mc";  
    
    std::string NameDBOracle2 = "oracle://cms_orcoff_prod/CMS_COND_20X_ECAL";
    std::string TagDBOracle2 = "EcalIntercalibConstants_startup_csa08_mc";  
 
       
    
     //---- location of the xml file ----      
    std::string calibFile = "/afs/cern.ch/user/a/amassiro/scratch0/CMSSW_2_1_2/src/CalibCalorimetry/CaloMiscalibTools/data/miscalib_barrel_startup_csa08.xml";
 
    
    
    if (argc == 1) {
        std::cout << "Help:"<<std::endl;
        std::cout << " 0-> Help"<< std::endl;
        std::cout << " 1-> Name of 1st Database. e.g. oracle://cms_orcoff_int2r/CMS_COND_ECAL"<< std::endl;
        std::cout << " 2-> Name of 1st Database Tag. e.g. EcalIntercalibConstants_startup_csa08_mc"<< std::endl;
        std::cout << " 3-> Location of the xml file, e.g. /afs/cern.ch/user/.../CMSSW_2_1_2/src/CalibCalorimetry/CaloMiscalibTools/data/miscalib_endcap_startup_csa08.xml. Make sure it is on your afs."<< std::endl;
        std::cout << " 4-> Name of 2nd Database. e.g. oracle://cms_orcoff_int2r/CMS_COND_ECAL"<< std::endl;
        std::cout << " 5-> Name of 2nd Database Tag. e.g. EcalIntercalibConstants_startup_csa08_mc"<< std::endl;
        std::cout << " 6-> EEetaStart"<< std::endl;
        std::cout << " 7-> EEetaEnd"<< std::endl;
        std::cout << " 8-> EEphiStart"<< std::endl;
        std::cout << " 9-> EEphiEnd"<< std::endl;
        std::cout << " 10-> power"<< std::endl;
        std::cout << " 11-> filename output"<< std::endl;
        std::cout << std::endl << std::endl << "Now working with default values" << std::endl;
    }
   
    if (argc == 2) {
        //---- Help option ----
        std::string testName = "--help";
        std::string testNameLoad = argv[1];
        if (argv[1] == testName){
            std::cout << "Help:"<<std::endl;
            std::cout << " 0-> Help. --help option"<< std::endl;
            std::cout << " 1-> Name of 1st Database. e.g. oracle://cms_orcoff_int2r/CMS_COND_ECAL"<< std::endl;
            std::cout << " 2-> Name of 1st Database Tag. e.g. EcalIntercalibConstants_startup_csa08_mc"<< std::endl;
            std::cout << " 3-> Location of the xml file, e.g. /afs/cern.ch/user/.../CMSSW_2_1_2/src/CalibCalorimetry/CaloMiscalibTools/data/miscalib_endcap_startup_csa08.xml. Make sure it is on your afs."<< std::endl;
            std::cout << " 4-> Name of 2nd Database. e.g. oracle://cms_orcoff_int2r/CMS_COND_ECAL"<< std::endl;
            std::cout << " 5-> Name of 2nd Database Tag. e.g. EcalIntercalibConstants_startup_csa08_mc"<< std::endl;
            std::cout << " 6-> EEetaStart"<< std::endl;
            std::cout << " 7-> EEetaEnd"<< std::endl;
            std::cout << " 8-> EEphiStart"<< std::endl;
            std::cout << " 9-> EEphiEnd"<< std::endl;
            std::cout << " 10-> power"<< std::endl;
            std::cout << " 11-> filename output"<< std::endl;
            exit (1);
        }
    }
    
    
    if (argc > 1){
        NameDBOracle1 = argv[1];
        TagDBOracle1 = argv[2];
        std::cout << "NameDBOracle1 = " << NameDBOracle1 << std::endl;
        std::cout << "TagDBOracle1 = " << TagDBOracle1 << std::endl;
        if (argc > 3){
            calibFile = argv[3];
            std::cout << "calibFile = " << calibFile << std::endl;
            if (argc > 4){
                NameDBOracle2 = argv[4];
                TagDBOracle2 = argv[5];
                std::cout << "NameDBOracle2 = " << NameDBOracle2 << std::endl;
                std::cout << "TagDBOracle2 = " << TagDBOracle2 << std::endl;
            }
        }
    }    
    
    if (argc > 6)
    {
        if (argc < 10)
        {
            std::cerr << "Too few (or too many) arguments passed" << std::endl ;
            exit (1) ;
        }
        EBetaStart = atoi (argv[6]) ;
        EBetaEnd = atoi (argv[7]) ;
        EBphiStart = atoi (argv[8]) ;
        EBphiEnd = atoi (argv[9]) ;
    }
    if (argc == 11) power = atoi (argv[10]) ;
    if (argc == 12) filename = argv[11] ;
    
    
      //---- export of the DB from oracle to sqlite db ----
   
   
    
    
    
    std::string Command2LineStr1 = "cmscond_export_iov -s " + NameDBOracle1 + " -d sqlite_file:Uno.db -D CondFormatsEcalObjects -t " + TagDBOracle1 + " -P /afs/cern.ch/cms/DB/conddb/";
   
    std::string Command2LineStr2 = "cmscond_export_iov -s " + NameDBOracle2 + " -d sqlite_file:Due.db -D CondFormatsEcalObjects -t " + TagDBOracle2 + " -P /afs/cern.ch/cms/DB/conddb/";
            
    std::cout << Command2LineStr1 << std::endl;
    std::cout << Command2LineStr2 << std::endl;
    
    
    gSystem->Exec(Command2LineStr1.c_str());
    
    //---- now the second set analysed through xml file ----
     gSystem->Exec(Command2LineStr2.c_str());
   
    
    
    
  
  std::string NameDB;
  std::string FileData;

  
   //----------------------------------
  //---- First Database Analyzed -----
  //----------------------------------
  NameDB = "sqlite_file:Uno.db";
  FileData = TagDBOracle1;
  CondIter<EcalIntercalibConstants> Iterator1;
  Iterator1.create(NameDB,FileData);
  
  
  
  //-----------------------------------
  //---- Second Database Analyzed -----
  //-----------------------------------
  
  NameDB = "sqlite_file:Due.db";  
  FileData = TagDBOracle2;
  CondIter<EcalIntercalibConstants> Iterator2;
  Iterator2.create(NameDB,FileData);

  //---------------------------------------------------------------------
  
  
      
  
  
  //-------------------------------------------------
  //---- Ottengo Mappe da entrambi gli Iterators ----
  //-------------------------------------------------

   
  const EcalIntercalibConstants* EBconstants1;
  EBconstants1 = Iterator1.next();
  EcalIntercalibConstantMap iEBcalibMap = EBconstants1->getMap () ;
 
  
  
  const EcalIntercalibConstants* EBconstants2;
  EBconstants2 = Iterator2.next();
  EcalIntercalibConstantMap iEBscalibMap = EBconstants2->getMap () ;

  
  
  
  
  
  
    //---- load form xml file ----
  
 
  
//   CaloMiscalibMapEcal EBscalibMap ;
//   EBscalibMap.prefillMap () ;
//   MiscalibReaderFromXMLEcalBarrel barrelreader (EBscalibMap) ;
//   if (!calibFile.empty ()) barrelreader.parseXMLMiscalibFile (calibFile) ;
//   EcalIntercalibConstants* EBconstants = new EcalIntercalibConstants (EBscalibMap.get ()) ;
//   EcalIntercalibConstantMap iEBscalibMap = EBconstants->getMap () ;  //MF prende i vecchi coeff

  
  
  
  
  
  
    //---- Xml way ----  
  
  
//   CaloMiscalibMapEcal EBscalibMap ;
//   EBscalibMap.prefillMap () ;
//   MiscalibReaderFromXMLEcalBarrel barrelreader (EBscalibMap) ;
//   if (!barrelfile.empty ()) barrelreader.parseXMLMiscalibFile (barrelfile) ;
//   EcalIntercalibConstants* EBconstants = 
//          new EcalIntercalibConstants (EBscalibMap.get ()) ;
//   EcalIntercalibConstantMap iEBscalibMap = EBconstants->getMap () ;  //MF prende i vecchi coeff
// 
//   //PG get the recalibration files for EB and EE
//   //PG -----------------------------------------
// 
//   CaloMiscalibMapEcal EBcalibMap ;
//   EBcalibMap.prefillMap () ;
//   MiscalibReaderFromXMLEcalBarrel calibBarrelreader (EBcalibMap) ;
//   if (!calibBarrelfile.empty ()) calibBarrelreader.parseXMLMiscalibFile (calibBarrelfile) ;
//   EcalIntercalibConstants* EBCconstants = 
//          new EcalIntercalibConstants (EBcalibMap.get ()) ;
//   EcalIntercalibConstantMap iEBcalibMap = EBCconstants->getMap () ;  //MF prende i vecchi coeff

  
  
  
  
  
  
  //PG fill the histograms
  //PG -------------------
  
  
   // ECAL barrel
  TH1F EBCompareCoeffDistr ("EBCompareCoeff","EBCompareCoeff",1000,0,2) ;
  TH2F EBCompareCoeffMap ("EBCompareCoeffMap","EBCompareCoeffMap",360,1,361,171,-85,86) ;
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
          
//           std::cout << " ieta = " << ieta  << " iphi = " << iphi << " iEBcalibMap --> " << *(iEBcalibMap.find (det.rawId ()));
//       std::cout << " iEBscalibMap --> " << *(iEBscalibMap.find (det.rawId ())) << std::endl;

          
          phiSum += factor ;
          phiSumSq += factor * factor ;
          N += 1. ;
          EBCompareCoeffDistr.Fill (factor) ;
          EBCompareCoeffMap.Fill (iphi,ieta,factor) ;
          EBCompareCoeffEtaTrend.Fill (ieta,factor) ;
          EBCompareCoeffEtaProfile.Fill (ieta,factor) ;
          if (abs(ieta) < 26) EBCompareCoeffDistr_M1.Fill (factor) ;
          else if (abs(ieta) < 46) EBCompareCoeffDistr_M2.Fill (factor) ;
          else if (abs(ieta) < 66) EBCompareCoeffDistr_M3.Fill (factor) ;
          else EBCompareCoeffDistr_M4.Fill (factor) ;
      } //PG loop over phi
  } //PG loop over eta

  // trend vs eta FIXME to be renormalized
  TH1D * EBEtaTrend = EBCompareCoeffMap.ProjectionY () ;
  // trend vs phi FIXME to be renormalized
  TH1D * EBPhiTrend = EBCompareCoeffMap.ProjectionX () ;


  TFile out (filename.c_str (),"recreate") ;
  
  EBCompareCoeffMap.GetXaxis()->SetTitle("i#phi");
  EBCompareCoeffMap.GetYaxis()->SetTitle("i#eta");
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
  
  
  
  
  //---- remove local database ----
  
  gSystem->Exec("rm Uno.db");
  gSystem->Exec("rm Due.db");

  
}
