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
    int EEphiStart = 0 ;
    int EEphiEnd = 360 ;

    std::string filename = "coeffcompareEE.root" ;
     
//     std::string NameDBOracle1 = "oracle://cms_orcoff_int2r/CMS_COND_ECAL";
//     std::string TagDBOracle1 = "EcalIntercalibConstants_startup_csa08_mc";  
//     
//     std::string NameDBOracle2 = "oracle://cms_orcoff_int2r/CMS_COND_ECAL";
//     std::string TagDBOracle2 = "EcalIntercalibConstants_inv_startup_csa08_mc";  

    std::string NameDBOracle1 = "oracle://cms_orcoff_prod/CMS_COND_20X_ECAL";
    std::string TagDBOracle1 = "EcalIntercalibConstants_phi_Zee_csa08_s156_mc";  
    
    std::string NameDBOracle2 = "oracle://cms_orcoff_prod/CMS_COND_20X_ECAL";
    std::string TagDBOracle2 = "EcalIntercalibConstants_startup_csa08_mc";  
    
    
    
    //---- location of the xml file ----
    std::string calibFile = "/afs/cern.ch/user/a/amassiro/scratch0/CMSSW_2_1_2/src/CalibCalorimetry/CaloMiscalibTools/data/miscalib_endcap_startup_csa08.xml";
 
    
    
    
    if (argc == 1) {
        std::cout << "Help:"<<std::endl;
        std::cout << " 0-> Help"<< std::endl;
        std::cout << " 1-> Name of 1st Database. e.g. oracle://cms_orcoff_int2r/CMS_COND_ECAL"<< std::endl;
        std::cout << " 2-> Name of 1st Database Tag. e.g. EcalIntercalibConstants_startup_csa08_mc"<< std::endl;
        std::cout << " 3-> Location of the xml file, e.g. /afs/cern.ch/user/.../CMSSW_2_1_2/src/CalibCalorimetry/CaloMiscalibTools/data/miscalib_endcap_startup_csa08.xml. Make sure it is on your afs."<< std::endl;
        std::cout << " 4-> Name of 2nd Database. e.g. oracle://cms_orcoff_int2r/CMS_COND_ECAL"<< std::endl;
        std::cout << " 5-> Name of 2nd Database Tag. e.g. EcalIntercalibConstants_startup_csa08_mc"<< std::endl;
        std::cout << " 6-> EEradStart"<< std::endl;
        std::cout << " 7-> EEradEnd"<< std::endl;
        std::cout << " 8-> EEphiStart"<< std::endl;
        std::cout << " 9-> EEphiEnd"<< std::endl;
        std::cout << " 10-> filename output"<< std::endl;
        
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
        std::cout << " 6-> EEradStart"<< std::endl;
        std::cout << " 7-> EEradEnd"<< std::endl;
        std::cout << " 8-> EEphiStart"<< std::endl;
        std::cout << " 9-> EEphiEnd"<< std::endl;
        std::cout << " 10-> filename output"<< std::endl;
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
        EEradStart = atoi (argv[6]) ;
        EEradEnd = atoi (argv[7]) ;
        EEphiStart = atoi (argv[8]) ;
        EEphiEnd = atoi (argv[9]) ;
    }
    if (argc == 11) filename = argv[10] ;

    
    
    
    //---- export of the DB from oracle to sqlite db ----
  
    
    std::string Command2LineStr1 = "cmscond_export_iov -s " + NameDBOracle1 + " -d sqlite_file:Uno.db -D CondFormatsEcalObjects -t " + TagDBOracle1 + " -P /afs/cern.ch/cms/DB/conddb/";
   
    std::string Command2LineStr2 = "cmscond_export_iov -s " + NameDBOracle2 + " -d sqlite_file:Due.db -D CondFormatsEcalObjects -t " + TagDBOracle2 + " -P /afs/cern.ch/cms/DB/conddb/";
            

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
  
  const EcalIntercalibConstants* EEconstants1;
  EEconstants1 = Iterator1.next();
  EcalIntercalibConstantMap iEEcalibMap = EEconstants1->getMap () ;
 
  
  
  const EcalIntercalibConstants* EEconstants2;
  EEconstants2 = Iterator2.next();
  EcalIntercalibConstantMap iEEscalibMap = EEconstants2->getMap () ;

  
  
  
  
  
  
  
  
  
  //---- load form xml file ----
  //---- name of the xml file defined at the beginning ----
  
//   CaloMiscalibMapEcal EEscalibMap ;
//   EEscalibMap.prefillMap () ;
//   MiscalibReaderFromXMLEcalEndcap endcapreader (EEscalibMap) ;
//   if (!calibFile.empty ()) endcapreader.parseXMLMiscalibFile (calibFile) ;
//   EcalIntercalibConstants* EEconstants = new EcalIntercalibConstants (EEscalibMap.get ()) ;
//   EcalIntercalibConstantMap iEEscalibMap = EEconstants->getMap () ;  //MF prende i vecchi coeff
    
  
  
  
  
  
  
  //---- Xml way ----  

  
  
//   CaloMiscalibMapEcal EEscalibMap ;
//   EEscalibMap.prefillMap () ;
//   MiscalibReaderFromXMLEcalEndcap endcapreader (EEscalibMap) ;
//   if (!endcapfile.empty ()) endcapreader.parseXMLMiscalibFile (endcapfile) ;
//   EcalIntercalibConstants* EEconstants = 
//           new EcalIntercalibConstants (EEscalibMap.get ()) ;
//   EcalIntercalibConstantMap iEEscalibMap = EEconstants->getMap () ;  //MF prende i vecchi coeff
// 
//   //PG get the recalibration files for EB and EE
//   //PG -----------------------------------------
// 
//   CaloMiscalibMapEcal EEcalibMap ;
//   EEcalibMap.prefillMap () ;
//   MiscalibReaderFromXMLEcalEndcap calibEndcapreader (EEcalibMap) ;
//   if (!calibEndcapfile.empty ()) calibEndcapreader.parseXMLMiscalibFile (calibEndcapfile) ;
//   EcalIntercalibConstants* EECconstants = 
//           new EcalIntercalibConstants (EEcalibMap.get ()) ;
//   EcalIntercalibConstantMap iEEcalibMap = EECconstants->getMap () ;  //MF prende i vecchi coeff
  
  
  
  
  
  
  
  
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
   
  TH1F EEPCompareCoeffDistr_R1 ("EEPCompareCoeff_R1","EEPCompareCoeff_R1",1000,0,2) ;
  TH1F EEPCompareCoeffDistr_R2 ("EEPCompareCoeff_R2","EEPCompareCoeff_R2",1000,0,2) ;
  TH1F EEPCompareCoeffDistr_R3 ("EEPCompareCoeff_R3","EEPCompareCoeff_R3",1000,0,2) ;
  TH1F EEPCompareCoeffDistr_R4 ("EEPCompareCoeff_R4","EEPCompareCoeff_R4",1000,0,2) ;
  TH1F EEPCompareCoeffDistr_R5 ("EEPCompareCoeff_R5","EEPCompareCoeff_R5",1000,0,2) ;

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
      double factor = *(iEEcalibMap.find (det.rawId ())) / 
                  *(iEEscalibMap.find (det.rawId ())) ;
      
//       std::cout << " iEEcalibMap rad = " << rad << " --> " << *(iEEcalibMap.find (det.rawId ()));
//       std::cout << " iEEscalibMap --> " << *(iEEscalibMap.find (det.rawId ())) << std::endl;  
      
      EEPCompareCoeffDistr.Fill (factor) ;
      EEPCompareCoeffMap.Fill (ix,iy,factor) ;
      EEPCompareCoeffEtaTrend.Fill (rad,factor) ;
      EEPCompareCoeffEtaProfile.Fill (rad,factor) ;
      if (abs(rad) < 22) continue ;
      else if (abs(rad) < 27) EEPCompareCoeffDistr_R1.Fill (factor) ;
      else if (abs(rad) < 32) EEPCompareCoeffDistr_R2.Fill (factor) ;
      else if (abs(rad) < 37) EEPCompareCoeffDistr_R3.Fill (factor) ;
      else if (abs(rad) < 42) EEPCompareCoeffDistr_R4.Fill (factor) ;
      else EEPCompareCoeffDistr_R5.Fill (factor) ;

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
      double factor = *(iEEcalibMap.find (det.rawId ())) *
                  *(iEEscalibMap.find (det.rawId ())) ;
      EEMCompareCoeffDistr.Fill (factor) ;
      EEMCompareCoeffMap.Fill (ix,iy,factor) ;
      EEMCompareCoeffEtaTrend.Fill (rad,factor) ;
      EEMCompareCoeffEtaProfile.Fill (rad,factor) ;
  } // ECAL endcap -
    
  TFile out (filename.c_str (),"recreate") ;
  EEMCompareCoeffMap.Write() ;
  EEMCompareCoeffDistr.Write() ;
  EEMCompareCoeffEtaTrend.Write () ;
  EEMCompareCoeffEtaProfile.Write () ;
  EEPCompareCoeffMap.Write() ;
  EEPCompareCoeffDistr.Write() ;
  EEPCompareCoeffEtaTrend.Write () ;
  EEPCompareCoeffEtaProfile.Write () ;
  EEPCompareCoeffDistr_R1.Write () ;
  EEPCompareCoeffDistr_R2.Write () ;
  EEPCompareCoeffDistr_R3.Write () ;
  EEPCompareCoeffDistr_R4.Write () ;
  EEPCompareCoeffDistr_R5.Write () ;
  out.Close() ;
  
  
  
  
  
  
  
  
  //---- remove local database ----
  
  gSystem->Exec("rm Uno.db");
  gSystem->Exec("rm Due.db");

  
   
}
