#include "DQM/CastorMonitor/interface/CastorHIMonitor.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

//***************************************************//
//********** CastorHIMonitor: *******************//
//********** Author: Dmytro Volyanskyy   ************//
//********** Date  : 14.10.2010 (first version) ******// 
//***************************************************//
////---- specific plots for HI runs  
////---- last revision: 

//==================================================================//
//======================= Constructor ==============================//
//==================================================================//
CastorHIMonitor::CastorHIMonitor() {
  doPerChannel_ = true;
  //  occThresh_ = 1;
  ievt_  =   0;
  module = -99;
  sector = -99;
  energy = -99;
  time   = -99;
  EtotalEM =-99;
  EtotalHAD =-99;
  EtotalCASTOR =-99;

}

//==================================================================//
//======================= Destructor ==============================//
//==================================================================//
CastorHIMonitor::~CastorHIMonitor(){
}

void CastorHIMonitor::reset(){
}


//==========================================================//
//========================= setup ==========================//
//==========================================================//

void CastorHIMonitor::setup(const edm::ParameterSet& ps, DQMStore* dbe){
  
  CastorBaseMonitor::setup(ps,dbe);
  baseFolder_ = rootFolder_+"CastorHIMonitor/EnergyUnits";

   if(fVerbosity>0) std::cout << "CastorHIMonitor::setup (start)" << std::endl;
  
  if ( ps.getUntrackedParameter<bool>("RecHitsPerChannel", false) ){
    doPerChannel_ = true;
  }
    
  ievt_=0; EtotalEM =0; EtotalHAD =0; EtotalCASTOR =0; 
 
 
 ////---- initialize the array energyInEachChannel
  for (int mod=0; mod<14; mod++) {
    for (int sec=0; sec<16; sec++){
        energyInEachChannel[mod][sec] = 0;
        energyTotalChannel[mod][sec] = 0;
    }
  }
  
///---- initialize the array energyHADsector,  energySectors
  for (int sec=0; sec<16; sec++){
        energyHADsector[sec] = 0;
        energySectors[sec] = 0;
        energyTotalSector[sec] =0;
      }
 
  if ( m_dbe !=NULL ) {    
 m_dbe->setCurrentFolder(baseFolder_);
 

////---- book MonitorElements
 

////----------------------------------------------------/////
////------------------ ENERGY --------------------------////
////---------------------------------------------------/////

 ////--- total energy in CASTOR per event 
 meEtotalCASTOR  =  m_dbe->book1D("CASTOR Absolute RecHit Energy per event","CASTOR Absolute RecHit Energy per event",400,0,2000);

 ////--- total energy in each CASTOR phi-sector 
 meEtotalSector  = m_dbe->book1D("CASTOR Total RecHit Energy in phi-sectors per run","Total RecHit Energy in phi-sectors per run",16,0,16);

 ////--- total EM energy per event
 meEtotalEM  =  m_dbe->book1D("CASTOR Total EM RecHit Energy per event","Total EM RecHit Energy per event",300,0,1500);

 ////--- total HAD energy per event 
 meEtotalHAD  =  m_dbe->book1D("CASTOR Total HAD RecHit Energy per event","Total HAD RecHit Energy per event",300,0,1500);

 ////--- total energy ratio EM/HAD per event
 meEtotalEMvsHAD  =  m_dbe->book1D("CASTOR Total Energy ratio EM to HAD per event","Total Energy ratio EM to HAD per event",50,0,5);


////---- energy in sectors
meEsector1  =  m_dbe->book1D("RecHit Energy in phi-sector1 vs z-module","RecHit Energy in phi-sector1 vs z-module",14,0,14);
meEsector2  =  m_dbe->book1D("RecHit Energy in phi-sector2 vs z-module","RecHit Energy in phi-sector2 vs z-module",14,0,14);
meEsector3  =  m_dbe->book1D("RecHit Energy in phi-sector3 vs z-module","RecHit Energy in phi-sector3 vs z-module",14,0,14);
meEsector4  =  m_dbe->book1D("RecHit Energy in phi-sector4 vs z-module","RecHit Energy in phi-sector4 vs z-module",14,0,14);
meEsector5  =  m_dbe->book1D("RecHit Energy in phi-sector5 vs z-module","RecHit Energy in phi-sector5 vs z-module",14,0,14);
meEsector6  =  m_dbe->book1D("RecHit Energy in phi-sector6 vs z-module","RecHit Energy in phi-sector6 vs z-module",14,0,14);
meEsector7  =  m_dbe->book1D("RecHit Energy in phi-sector7 vs z-module","RecHit Energy in phi-sector7 vs z-module",14,0,14);
meEsector8  =  m_dbe->book1D("RecHit Energy in phi-sector8 vs z-module","RecHit Energy in phi-sector8 vs z-module",14,0,14);
meEsector9  =  m_dbe->book1D("RecHit Energy in phi-sector9 vs z-module","RecHit Energy in phi-sector9 vs z-module",14,0,14);
meEsector10 =  m_dbe->book1D("RecHit Energy in phi-sector10 vs z-module","RecHit Energy in phi-sector10 vs z-module",14,0,14);
meEsector11 =  m_dbe->book1D("RecHit Energy in phi-sector11 vs z-module","RecHit Energy in phi-sector11 vs z-module",14,0,14);
meEsector12 =  m_dbe->book1D("RecHit Energy in phi-sector12 vs z-module","RecHit Energy in phi-sector12 vs z-module",14,0,14);
meEsector13 =  m_dbe->book1D("RecHit Energy in phi-sector13 vs z-module","RecHit Energy in phi-sector13 vs z-module",14,0,14);
meEsector14 =  m_dbe->book1D("RecHit Energy in phi-sector14 vs z-module","RecHit Energy in phi-sector14 vs z-module",14,0,14);
meEsector15 =  m_dbe->book1D("RecHit Energy in phi-sector15 vs z-module","RecHit Energy in phi-sector15 vs z-module",14,0,14);
meEsector16 =  m_dbe->book1D("RecHit Energy in phi-sector16 vs z-module","RecHit Energy in phi-sector16 vs z-module",14,0,14);



 m_dbe->setCurrentFolder(baseFolder_+"/furtherPlots");

 ////---- event number
meEVT_ = m_dbe->bookInt("HI Event Number");

////---- energy in modules
meEmodule1  =  m_dbe->book1D("RecHit Energy in z-module1 vs phi-sector","RecHit Energy in z-module1 vs phi-sector",16,0,16);
meEmodule2  =  m_dbe->book1D("RecHit Energy in z-module2 vs phi-sector","RecHit Energy in z-module2 vs phi-sector",16,0,16);
meEmodule3  =  m_dbe->book1D("RecHit Energy in z-module3 vs phi-sector","RecHit Energy in z-module3 vs phi-sector",16,0,16);
meEmodule4  =  m_dbe->book1D("RecHit Energy in z-module4 vs phi-sector","RecHit Energy in z-module4 vs phi-sector",16,0,16);
meEmodule5  =  m_dbe->book1D("RecHit Energy in z-module5 vs phi-sector","RecHit Energy in z-module5 vs phi-sector",16,0,16);
meEmodule6  =  m_dbe->book1D("RecHit Energy in z-module6 vs phi-sector","RecHit Energy in z-module6 vs phi-sector",16,0,16);
meEmodule7  =  m_dbe->book1D("RecHit Energy in z-module7 vs phi-sector","RecHit Energy in z-module7 vs phi-sector",16,0,16);
meEmodule8  =  m_dbe->book1D("RecHit Energy in z-module8 vs phi-sector","RecHit Energy in z-module8 vs phi-sector",16,0,16);
meEmodule9  =  m_dbe->book1D("RecHit Energy in z-module9 vs phi-sector","RecHit Energy in z-module9 vs phi-sector",16,0,16);
meEmodule10 =  m_dbe->book1D("RecHit Energy in z-module10 vs phi-sector","RecHit Energy in z-module10 vs phi-sector",16,0,16);
meEmodule11 =  m_dbe->book1D("RecHit Energy in z-module11 vs phi-sector","RecHit Energy in z-module11 vs phi-sector",16,0,16);
meEmodule12 =  m_dbe->book1D("RecHit Energy in z-module12 vs phi-sector","RecHit Energy in z-module12 vs phi-sector",16,0,16);
meEmodule13 =  m_dbe->book1D("RecHit Energy in z-module13 vs phi-sector","RecHit Energy in z-module13 vs phi-sector",16,0,16);
meEmodule14 =  m_dbe->book1D("RecHit Energy in z-module14 vs phi-sector","RecHit Energy in z-module14 vs phi-sector",16,0,16);

 double EmaxSector=800;
 double NbinsSector=200;

////---- energy in EM sectors (sum in modules 1-2)
meEsectorEM1  =  m_dbe->book1D("RecHit Energy in EM phi-sector1","RecHit Energy in EM phi-sector1",NbinsSector,0,EmaxSector);
meEsectorEM2  =  m_dbe->book1D("RecHit Energy in EM phi-sector2","RecHit Energy in EM phi-sector2",NbinsSector,0,EmaxSector);
meEsectorEM3  =  m_dbe->book1D("RecHit Energy in EM phi-sector3","RecHit Energy in EM phi-sector3",NbinsSector,0,EmaxSector);
meEsectorEM4  =  m_dbe->book1D("RecHit Energy in EM phi-sector4","RecHit Energy in EM phi-sector4",NbinsSector,0,EmaxSector);
meEsectorEM5  =  m_dbe->book1D("RecHit Energy in EM phi-sector5","RecHit Energy in EM phi-sector5",NbinsSector,0,EmaxSector);
meEsectorEM6  =  m_dbe->book1D("RecHit Energy in EM phi-sector6","RecHit Energy in EM phi-sector6",NbinsSector,0,EmaxSector);
meEsectorEM7  =  m_dbe->book1D("RecHit Energy in EM phi-sector7","RecHit Energy in EM phi-sector7",NbinsSector,0,EmaxSector);
meEsectorEM8  =  m_dbe->book1D("RecHit Energy in EM phi-sector8","RecHit Energy in EM phi-sector8",NbinsSector,0,EmaxSector);
meEsectorEM9  =  m_dbe->book1D("RecHit Energy in EM phi-sector9","RecHit Energy in EM phi-sector9",NbinsSector,0,EmaxSector);
meEsectorEM10  =  m_dbe->book1D("RecHit Energy in EM phi-sector10","RecHit Energy in EM phi-sector10",NbinsSector,0,EmaxSector);
meEsectorEM11  =  m_dbe->book1D("RecHit Energy in EM phi-sector11","RecHit Energy in EM phi-sector11",NbinsSector,0,EmaxSector);
meEsectorEM12  =  m_dbe->book1D("RecHit Energy in EM phi-sector12","RecHit Energy in EM phi-sector12",NbinsSector,0,EmaxSector);
meEsectorEM13  =  m_dbe->book1D("RecHit Energy in EM phi-sector13","RecHit Energy in EM phi-sector13",NbinsSector,0,EmaxSector);
meEsectorEM14  =  m_dbe->book1D("RecHit Energy in EM phi-sector14","RecHit Energy in EM phi-sector14",NbinsSector,0,EmaxSector);
meEsectorEM15  =  m_dbe->book1D("RecHit Energy in EM phi-sector15","RecHit Energy in EM phi-sector15",NbinsSector,0,EmaxSector);
meEsectorEM16  =  m_dbe->book1D("RecHit Energy in EM phi-sector16","RecHit Energy in EM phi-sector16",NbinsSector,0,EmaxSector);

////---- energy in HAD sectors (sum in modules 3-14)
meEsectorHAD1  =  m_dbe->book1D("RecHit Energy in HAD phi-sector1","RecHit Energy in HAD phi-sector1",NbinsSector,0,EmaxSector);
meEsectorHAD2  =  m_dbe->book1D("RecHit Energy in HAD phi-sector2","RecHit Energy in HAD phi-sector2",NbinsSector,0,EmaxSector);
meEsectorHAD3  =  m_dbe->book1D("RecHit Energy in HAD phi-sector3","RecHit Energy in HAD phi-sector3",NbinsSector,0,EmaxSector);
meEsectorHAD4  =  m_dbe->book1D("RecHit Energy in HAD phi-sector4","RecHit Energy in HAD phi-sector4",NbinsSector,0,EmaxSector);
meEsectorHAD5  =  m_dbe->book1D("RecHit Energy in HAD phi-sector5","RecHit Energy in HAD phi-sector5",NbinsSector,0,EmaxSector);
meEsectorHAD6  =  m_dbe->book1D("RecHit Energy in HAD phi-sector6","RecHit Energy in HAD phi-sector6",NbinsSector,0,EmaxSector);
meEsectorHAD7  =  m_dbe->book1D("RecHit Energy in HAD phi-sector7","RecHit Energy in HAD phi-sector7",NbinsSector,0,EmaxSector);
meEsectorHAD8  =  m_dbe->book1D("RecHit Energy in HAD phi-sector8","RecHit Energy in HAD phi-sector8",NbinsSector,0,EmaxSector);
meEsectorHAD9  =  m_dbe->book1D("RecHit Energy in HAD phi-sector9","RecHit Energy in HAD phi-sector9",NbinsSector,0,EmaxSector);
meEsectorHAD10  =  m_dbe->book1D("RecHit Energy in HAD phi-sector10","RecHit Energy in HAD phi-sector10",NbinsSector,0,EmaxSector);
meEsectorHAD11  =  m_dbe->book1D("RecHit Energy in HAD phi-sector11","RecHit Energy in HAD phi-sector11",NbinsSector,0,EmaxSector);
meEsectorHAD12  =  m_dbe->book1D("RecHit Energy in HAD phi-sector12","RecHit Energy in HAD phi-sector12",NbinsSector,0,EmaxSector);
meEsectorHAD13  =  m_dbe->book1D("RecHit Energy in HAD phi-sector13","RecHit Energy in HAD phi-sector13",NbinsSector,0,EmaxSector);
meEsectorHAD14  =  m_dbe->book1D("RecHit Energy in HAD phi-sector14","RecHit Energy in HAD phi-sector14",NbinsSector,0,EmaxSector);
meEsectorHAD15  =  m_dbe->book1D("RecHit Energy in HAD phi-sector15","RecHit Energy in HAD phi-sector15",NbinsSector,0,EmaxSector);
meEsectorHAD16  =  m_dbe->book1D("RecHit Energy in HAD phi-sector16","RecHit Energy in HAD phi-sector16",NbinsSector,0,EmaxSector);


////---- energy ratio EM/HAD in sectors
meEsectorEMvsHAD1  =  m_dbe->book1D("Ratio E_EM to E_HAD phi-sector1","Ratio E_EM to E_HAD phi-sector1",50,0,5);
meEsectorEMvsHAD2  =  m_dbe->book1D("Ratio E_EM to E_HAD phi-sector2","Ratio E_EM to E_HAD phi-sector2",50,0,5);
meEsectorEMvsHAD3  =  m_dbe->book1D("Ratio E_EM to E_HAD phi-sector3","Ratio E_EM to E_HAD phi-sector3",50,0,5);
meEsectorEMvsHAD4  =  m_dbe->book1D("Ratio E_EM to E_HAD phi-sector4","Ratio E_EM to E_HAD phi-sector4",50,0,5);
meEsectorEMvsHAD5  =  m_dbe->book1D("Ratio E_EM to E_HAD phi-sector5","Ratio E_EM to E_HAD phi-sector5",50,0,5);
meEsectorEMvsHAD6  =  m_dbe->book1D("Ratio E_EM to E_HAD phi-sector6","Ratio E_EM to E_HAD phi-sector6",50,0,5);
meEsectorEMvsHAD7  =  m_dbe->book1D("Ratio E_EM to E_HAD phi-sector7","Ratio E_EM to E_HAD phi-sector7",50,0,5);
meEsectorEMvsHAD8  =  m_dbe->book1D("Ratio E_EM to E_HAD phi-sector8","Ratio E_EM to E_HAD phi-sector8",50,0,5);
meEsectorEMvsHAD9  =  m_dbe->book1D("Ratio E_EM to E_HAD phi-sector9","Ratio E_EM to E_HAD phi-sector9",50,0,5);
meEsectorEMvsHAD10  =  m_dbe->book1D("Ratio E_EM to E_HAD phi-sector10","Ratio E_EM to E_HAD phi-sector10",50,0,5);
meEsectorEMvsHAD11  =  m_dbe->book1D("Ratio E_EM to E_HAD phi-sector11","Ratio E_EM to E_HAD phi-sector11",50,0,5);
meEsectorEMvsHAD12  =  m_dbe->book1D("Ratio E_EM to E_HAD phi-sector12","Ratio E_EM to E_HAD phi-sector12",50,0,5);
meEsectorEMvsHAD13  =  m_dbe->book1D("Ratio E_EM to E_HAD phi-sector13","Ratio E_EM to E_HAD phi-sector13",50,0,5);
meEsectorEMvsHAD14  =  m_dbe->book1D("Ratio E_EM to E_HAD phi-sector14","Ratio E_EM to E_HAD phi-sector14",50,0,5);
meEsectorEMvsHAD15  =  m_dbe->book1D("Ratio E_EM to E_HAD phi-sector15","Ratio E_EM to E_HAD phi-sector15",50,0,5);
meEsectorEMvsHAD16  =  m_dbe->book1D("Ratio E_EM to E_HAD phi-sector16","Ratio E_EM to E_HAD phi-sector16",50,0,5);




 /*

////----------------------------------------------------/////
////------------------ CHARGE --------------------------////
////---------------------------------------------------/////

 m_dbe->setCurrentFolder(rootFolder_+"CastorHIMonitor/Charge_fC");

////--- total charge in CASTOR per event 
 meChargeTotalCASTOR  =  m_dbe->book1D("CASTOR Total Charge per event","CASTOR Total Charge per event",100,0,400);

 ////--- total charge in each CASTOR phi-sector 
 meChargeTotalSector  = m_dbe->book1D("Total Charge in phi-sectors per event","Total Charge in phi-sectors per event",16,0,16);

 ////--- total EM charge per event
 meChargeTotalEM  =  m_dbe->book1D("Total EM Charge per event","Total EM Charge per event",100,0,400);

 ////--- total HAD charge per event 
 meChargeTotalHAD  =  m_dbe->book1D("Total HAD Charge per event","Total HAD Charge per event",100,0,400);

 ////--- total charge ratio EM/HAD per event
 meChargeTotalEMvsHAD  =  m_dbe->book1D("Total Charge ratio EM to HAD per event","Total Charge ratio EM to HAD per event",50,0,5);

////---- charge in sectors
meChargeSector1  =  m_dbe->book1D("Charge in phi-sector1 vs z-module","Charge in phi-sector1 vs z-module",14,0,14);
meChargeSector2  =  m_dbe->book1D("Charge in phi-sector2 vs z-module","Charge in phi-sector2 vs z-module",14,0,14);
meChargeSector3  =  m_dbe->book1D("Charge in phi-sector3 vs z-module","Charge in phi-sector3 vs z-module",14,0,14);
meChargeSector4  =  m_dbe->book1D("Charge in phi-sector4 vs z-module","Charge in phi-sector4 vs z-module",14,0,14);
meChargeSector5  =  m_dbe->book1D("Charge in phi-sector5 vs z-module","Charge in phi-sector5 vs z-module",14,0,14);
meChargeSector6  =  m_dbe->book1D("Charge in phi-sector6 vs z-module","Charge in phi-sector6 vs z-module",14,0,14);
meChargeSector7  =  m_dbe->book1D("Charge in phi-sector7 vs z-module","Charge in phi-sector7 vs z-module",14,0,14);
meChargeSector8  =  m_dbe->book1D("Charge in phi-sector8 vs z-module","Charge in phi-sector8 vs z-module",14,0,14);
meChargeSector9  =  m_dbe->book1D("Charge in phi-sector9 vs z-module","Charge in phi-sector9 vs z-module",14,0,14);
meChargeSector10 =  m_dbe->book1D("Charge in phi-sector10 vs z-module","Charge in phi-sector10 vs z-module",14,0,14);
meChargeSector11 =  m_dbe->book1D("Charge in phi-sector11 vs z-module","Charge in phi-sector11 vs z-module",14,0,14);
meChargeSector12 =  m_dbe->book1D("Charge in phi-sector12 vs z-module","Charge in phi-sector12 vs z-module",14,0,14);
meChargeSector13 =  m_dbe->book1D("Charge in phi-sector13 vs z-module","Charge in phi-sector13 vs z-module",14,0,14);
meChargeSector14 =  m_dbe->book1D("Charge in phi-sector14 vs z-module","Charge in phi-sector14 vs z-module",14,0,14);
meChargeSector15 =  m_dbe->book1D("Charge in phi-sector15 vs z-module","Charge in phi-sector15 vs z-module",14,0,14);
meChargeSector16 =  m_dbe->book1D("Charge in phi-sector16 vs z-module","Charge in phi-sector16 vs z-module",14,0,14);

 m_dbe->setCurrentFolder(rootFolder_+"CastorHIMonitor/Charge_fC/furtherPlots");

////---- charge in modules
meChargeModule1  =  m_dbe->book1D("Charge in z-module1 vs phi-sector","Charge in z-module1 vs phi-sector",16,0,16);
meChargeModule2  =  m_dbe->book1D("Charge in z-module2 vs phi-sector","Charge in z-module2 vs phi-sector",16,0,16);
meChargeModule3  =  m_dbe->book1D("Charge in z-module3 vs phi-sector","Charge in z-module3 vs phi-sector",16,0,16);
meChargeModule4  =  m_dbe->book1D("Charge in z-module4 vs phi-sector","Charge in z-module4 vs phi-sector",16,0,16);
meChargeModule5  =  m_dbe->book1D("Charge in z-module5 vs phi-sector","Charge in z-module5 vs phi-sector",16,0,16);
meChargeModule6  =  m_dbe->book1D("Charge in z-module6 vs phi-sector","Charge in z-module6 vs phi-sector",16,0,16);
meChargeModule7  =  m_dbe->book1D("Charge in z-module7 vs phi-sector","Charge in z-module7 vs phi-sector",16,0,16);
meChargeModule8  =  m_dbe->book1D("Charge in z-module8 vs phi-sector","Charge in z-module8 vs phi-sector",16,0,16);
meChargeModule9  =  m_dbe->book1D("Charge in z-module9 vs phi-sector","Charge in z-module9 vs phi-sector",16,0,16);
meChargeModule10 =  m_dbe->book1D("Charge in z-module10 vs phi-sector","Charge in z-module10 vs phi-sector",16,0,16);
meChargeModule11 =  m_dbe->book1D("Charge in z-module11 vs phi-sector","Charge in z-module11 vs phi-sector",16,0,16);
meChargeModule12 =  m_dbe->book1D("Charge in z-module12 vs phi-sector","Charge in z-module12 vs phi-sector",16,0,16);
meChargeModule13 =  m_dbe->book1D("Charge in z-module13 vs phi-sector","Charge in z-module13 vs phi-sector",16,0,16);
meChargeModule14 =  m_dbe->book1D("Charge in z-module14 vs phi-sector","Charge in z-module14 vs phi-sector",16,0,16);


////---- charge in EM sectors (sum in modules 1-2)
meChargeSectorEM1  =  m_dbe->book1D("Charge in EM phi-sector1","Charge in EM phi-sector1",50,0,200);
meChargeSectorEM2  =  m_dbe->book1D("Charge in EM phi-sector2","Charge in EM phi-sector2",50,0,200);
meChargeSectorEM3  =  m_dbe->book1D("Charge in EM phi-sector3","Charge in EM phi-sector3",50,0,200);
meChargeSectorEM4  =  m_dbe->book1D("Charge in EM phi-sector4","Charge in EM phi-sector4",50,0,200);
meChargeSectorEM5  =  m_dbe->book1D("Charge in EM phi-sector5","Charge in EM phi-sector5",50,0,200);
meChargeSectorEM6  =  m_dbe->book1D("Charge in EM phi-sector6","Charge in EM phi-sector6",50,0,200);
meChargeSectorEM7  =  m_dbe->book1D("Charge in EM phi-sector7","Charge in EM phi-sector7",50,0,200);
meChargeSectorEM8  =  m_dbe->book1D("Charge in EM phi-sector8","Charge in EM phi-sector8",50,0,200);
meChargeSectorEM9  =  m_dbe->book1D("Charge in EM phi-sector9","Charge in EM phi-sector9",50,0,200);
meChargeSectorEM10  =  m_dbe->book1D("Charge in EM phi-sector10","Charge in EM phi-sector10",50,0,200);
meChargeSectorEM11  =  m_dbe->book1D("Charge in EM phi-sector11","Charge in EM phi-sector11",50,0,200);
meChargeSectorEM12  =  m_dbe->book1D("Charge in EM phi-sector12","Charge in EM phi-sector12",50,0,200);
meChargeSectorEM13  =  m_dbe->book1D("Charge in EM phi-sector13","Charge in EM phi-sector13",50,0,200);
meChargeSectorEM14  =  m_dbe->book1D("Charge in EM phi-sector14","Charge in EM phi-sector14",50,0,200);
meChargeSectorEM15  =  m_dbe->book1D("Charge in EM phi-sector15","Charge in EM phi-sector15",50,0,200);
meChargeSectorEM16  =  m_dbe->book1D("Charge in EM phi-sector16","Charge in EM phi-sector16",50,0,200);

////---- charge in HAD phi-sectors (sum in modules 3-14)
meChargeSectorHAD1  =  m_dbe->book1D("Charge in HAD phi-sector1","Charge in HAD phi-sector1",50,0,200);
meChargeSectorHAD2  =  m_dbe->book1D("Charge in HAD phi-sector2","Charge in HAD phi-sector2",50,0,200);
meChargeSectorHAD3  =  m_dbe->book1D("Charge in HAD phi-sector3","Charge in HAD phi-sector3",50,0,200);
meChargeSectorHAD4  =  m_dbe->book1D("Charge in HAD phi-sector4","Charge in HAD phi-sector4",50,0,200);
meChargeSectorHAD5  =  m_dbe->book1D("Charge in HAD phi-sector5","Charge in HAD phi-sector5",50,0,200);
meChargeSectorHAD6  =  m_dbe->book1D("Charge in HAD phi-sector6","Charge in HAD phi-sector6",50,0,200);
meChargeSectorHAD7  =  m_dbe->book1D("Charge in HAD phi-sector7","Charge in HAD phi-sector7",50,0,200);
meChargeSectorHAD8  =  m_dbe->book1D("Charge in HAD phi-sector8","Charge in HAD phi-sector8",50,0,200);
meChargeSectorHAD9  =  m_dbe->book1D("Charge in HAD phi-sector9","Charge in HAD phi-sector9",50,0,200);
meChargeSectorHAD10  =  m_dbe->book1D("Charge in HAD phi-sector10","Charge in HAD phi-sector10",50,0,200);
meChargeSectorHAD11  =  m_dbe->book1D("Charge in HAD phi-sector11","Charge in HAD phi-sector11",50,0,200);
meChargeSectorHAD12  =  m_dbe->book1D("Charge in HAD phi-sector12","Charge in HAD phi-sector12",50,0,200);
meChargeSectorHAD13  =  m_dbe->book1D("Charge in HAD phi-sector13","Charge in HAD phi-sector13",50,0,200);
meChargeSectorHAD14  =  m_dbe->book1D("Charge in HAD phi-sector14","Charge in HAD phi-sector14",50,0,200);
meChargeSectorHAD15  =  m_dbe->book1D("Charge in HAD phi-sector15","Charge in HAD phi-sector15",50,0,200);
meChargeSectorHAD16  =  m_dbe->book1D("Charge in HAD phi-sector16","Charge in HAD phi-sector16",50,0,200);



////---- charge ratio EM/HAD in sectors
meChargeSectorEMvsHAD1  =  m_dbe->book1D("Ratio E_EM to E_HAD phi-sector1","Ratio E_EM to E_HAD phi-sector1",50,0,5);
meChargeSectorEMvsHAD2  =  m_dbe->book1D("Ratio E_EM to E_HAD phi-sector2","Ratio E_EM to E_HAD phi-sector2",50,0,5);
meChargeSectorEMvsHAD3  =  m_dbe->book1D("Ratio E_EM to E_HAD phi-sector3","Ratio E_EM to E_HAD phi-sector3",50,0,5);
meChargeSectorEMvsHAD4  =  m_dbe->book1D("Ratio E_EM to E_HAD phi-sector4","Ratio E_EM to E_HAD phi-sector4",50,0,5);
meChargeSectorEMvsHAD5  =  m_dbe->book1D("Ratio E_EM to E_HAD phi-sector5","Ratio E_EM to E_HAD phi-sector5",50,0,5);
meChargeSectorEMvsHAD6  =  m_dbe->book1D("Ratio E_EM to E_HAD phi-sector6","Ratio E_EM to E_HAD phi-sector6",50,0,5);
meChargeSectorEMvsHAD7  =  m_dbe->book1D("Ratio E_EM to E_HAD phi-sector7","Ratio E_EM to E_HAD phi-sector7",50,0,5);
meChargeSectorEMvsHAD8  =  m_dbe->book1D("Ratio E_EM to E_HAD phi-sector8","Ratio E_EM to E_HAD phi-sector8",50,0,5);
meChargeSectorEMvsHAD9  =  m_dbe->book1D("Ratio E_EM to E_HAD phi-sector9","Ratio E_EM to E_HAD phi-sector9",50,0,5);
meChargeSectorEMvsHAD10  =  m_dbe->book1D("Ratio E_EM to E_HAD phi-sector10","Ratio E_EM to E_HAD phi-sector10",50,0,5);
meChargeSectorEMvsHAD11  =  m_dbe->book1D("Ratio E_EM to E_HAD phi-sector11","Ratio E_EM to E_HAD phi-sector11",50,0,5);
meChargeSectorEMvsHAD12  =  m_dbe->book1D("Ratio E_EM to E_HAD phi-sector12","Ratio E_EM to E_HAD phi-sector12",50,0,5);
meChargeSectorEMvsHAD13  =  m_dbe->book1D("Ratio E_EM to E_HAD phi-sector13","Ratio E_EM to E_HAD phi-sector13",50,0,5);
meChargeSectorEMvsHAD14  =  m_dbe->book1D("Ratio E_EM to E_HAD phi-sector14","Ratio E_EM to E_HAD phi-sector14",50,0,5);
meChargeSectorEMvsHAD15  =  m_dbe->book1D("Ratio E_EM to E_HAD phi-sector15","Ratio E_EM to E_HAD phi-sector15",50,0,5);
meChargeSectorEMvsHAD16  =  m_dbe->book1D("Ratio E_EM to E_HAD phi-sector16","Ratio E_EM to E_HAD phi-sector16",50,0,5);



 */







  }//-- end of if

  else{
  if(fVerbosity>0) std::cout << "CastorHIMonitor::setup - NO DQMStore service" << std::endl; 
 }

  if(fVerbosity>0) std::cout << "CastorHIMonitor::setup (end)" << std::endl;

  return;
}






//==========================================================//
//================== processEvent ==========================//
//==========================================================//


void CastorHIMonitor::processEvent(const CastorRecHitCollection& castorHits, const CastorDigiCollection& castorDigis, const CastorDbService& cond ){  

  if(fVerbosity>0) std::cout << "==>CastorHIMonitor::processEvent !!!"<< std::endl;

  if(!m_dbe) { 
    if(fVerbosity>0) std::cout <<"CastorHIMonitor::processEvent => DQMStore is not instantiated !!!"<<std::endl;  
    return; 
  }

  ////---- fill the event number
  meEVT_->Fill(ievt_);

  ////---- initialize these variables for each new event
  EtotalEM =0; EtotalHAD =0; EtotalCASTOR =0;

  ////---- initialize the array energyInEachChannel for each new event
       for (int mod=0; mod<14; mod++) {
        for (int sec=0; sec<16; sec++) {
          energyInEachChannel[mod][sec] = 0;
       }
     }

  ////---- initialize the array energyHADsector for each new event
       for (int sec=0; sec<16; sec++)
     energyHADsector[sec] = 0;
   


  ////---- define the iterator
  CastorRecHitCollection::const_iterator CASTORiter;
  if (showTiming)  { cpu_timer.reset(); cpu_timer.start(); } 

  //********************************************************//
  //************ working with RecHits *********************//
  //*******************************************************//
 
  if(castorHits.size()>0)
  {    
 
     if(fVerbosity>0) std::cout << "==>CastorHIMonitor::processEvent: castorHits.size()>0 !!!" << std::endl; 

    ////---- loop over all hits in an event
    for (CASTORiter=castorHits.begin(); CASTORiter!=castorHits.end(); ++CASTORiter) { 
	
      ////---- get CastorID 
      HcalCastorDetId id(CASTORiter->detid().rawId());
      ////---- get module and sector
      module = (int)id.module(); 
      sector = (int)id.sector(); 
      //zside  = (int)id.zside(); 
      ////---- get energy and time for every hit:
      energy = CASTORiter->energy();    
      time   = CASTORiter->time();
      
      if(fVerbosity>0) 
      std::cout<<"CastorHIMonitor==> module:"<< module << " sector:"<< sector << " energy:" << energy<<std::endl;

      ////--- don't deal with negative values           
      if (energy<0) energy=0;

      ////--- fill the array energyInEachChannel for a particular event
      energyInEachChannel[module-1][sector-1] = energy; 
   
      ////--- fill the array energyTotalChannel 
      energyTotalChannel[module-1][sector-1] += energy; 


      }
      //-- end of the loop over all hits
      

      //-----------------------------------//
      //------- fill energy in sectors ----//
      //-- energy distribution in depth --//
      //----------------------------------//
      for (int mod=0; mod<14;mod++){
       meEsector1->Fill(mod,energyInEachChannel[mod][0]);
       meEsector2->Fill(mod,energyInEachChannel[mod][1]);
       meEsector3->Fill(mod,energyInEachChannel[mod][2]);
       meEsector4->Fill(mod,energyInEachChannel[mod][3]);
       meEsector5->Fill(mod,energyInEachChannel[mod][4]);
       meEsector6->Fill(mod,energyInEachChannel[mod][5]);
       meEsector7->Fill(mod,energyInEachChannel[mod][6]);
       meEsector8->Fill(mod,energyInEachChannel[mod][7]);
       meEsector9->Fill(mod,energyInEachChannel[mod][8]);
       meEsector10->Fill(mod,energyInEachChannel[mod][9]);
       meEsector11->Fill(mod,energyInEachChannel[mod][10]);
       meEsector12->Fill(mod,energyInEachChannel[mod][11]);
       meEsector13->Fill(mod,energyInEachChannel[mod][12]);
       meEsector14->Fill(mod,energyInEachChannel[mod][13]);
       meEsector15->Fill(mod,energyInEachChannel[mod][14]);
       meEsector16->Fill(mod,energyInEachChannel[mod][15]);
      }


      //------------------------------------------//
      //-- fill energy in modules    --//
      //-- energy distribution in each of the rings vs phi-sector --//
      //---------------------------//
      for (int sec=0; sec<16;sec++){
       meEmodule1->Fill(sec,energyInEachChannel[0][sec]);
       meEmodule2->Fill(sec,energyInEachChannel[1][sec]);
       meEmodule3->Fill(sec,energyInEachChannel[2][sec]);
       meEmodule4->Fill(sec,energyInEachChannel[3][sec]);
       meEmodule5->Fill(sec,energyInEachChannel[4][sec]);
       meEmodule6->Fill(sec,energyInEachChannel[5][sec]);
       meEmodule7->Fill(sec,energyInEachChannel[6][sec]);
       meEmodule8->Fill(sec,energyInEachChannel[7][sec]);
       meEmodule9->Fill(sec,energyInEachChannel[8][sec]);
       meEmodule10->Fill(sec,energyInEachChannel[9][sec]);
       meEmodule11->Fill(sec,energyInEachChannel[10][sec]);
       meEmodule12->Fill(sec,energyInEachChannel[11][sec]);
       meEmodule13->Fill(sec,energyInEachChannel[12][sec]);
       meEmodule14->Fill(sec,energyInEachChannel[13][sec]);
      }



      //-------------------------------//
      //-- fill energy in EM sectors --//
      //-------------------------------//
      meEsectorEM1->Fill(energyInEachChannel[0][0]+energyInEachChannel[1][0]);
      meEsectorEM2->Fill(energyInEachChannel[0][1]+energyInEachChannel[1][1]); 
      meEsectorEM3->Fill(energyInEachChannel[0][2]+energyInEachChannel[1][2]);
      meEsectorEM4->Fill(energyInEachChannel[0][3]+energyInEachChannel[1][3]);
      meEsectorEM5->Fill(energyInEachChannel[0][4]+energyInEachChannel[1][4]);
      meEsectorEM6->Fill(energyInEachChannel[0][5]+energyInEachChannel[1][5]);
      meEsectorEM7->Fill(energyInEachChannel[0][6]+energyInEachChannel[1][6]);
      meEsectorEM8->Fill(energyInEachChannel[0][7]+energyInEachChannel[1][7]);
      meEsectorEM9->Fill(energyInEachChannel[0][8]+energyInEachChannel[1][8]);
      meEsectorEM10->Fill(energyInEachChannel[0][9]+energyInEachChannel[1][9]);
      meEsectorEM11->Fill(energyInEachChannel[0][10]+energyInEachChannel[1][10]);
      meEsectorEM12->Fill(energyInEachChannel[0][11]+energyInEachChannel[1][11]);
      meEsectorEM13->Fill(energyInEachChannel[0][12]+energyInEachChannel[1][12]);
      meEsectorEM14->Fill(energyInEachChannel[0][13]+energyInEachChannel[1][13]);
      meEsectorEM15->Fill(energyInEachChannel[0][14]+energyInEachChannel[1][14]);
      meEsectorEM16->Fill(energyInEachChannel[0][15]+energyInEachChannel[1][15]);



      //-------------------------------//
      //-- fill energy in HAD sectors --//
      //-------------------------------//

       for(int sec=0; sec<16;sec++){
      
      ////--- sum over all modules
      energyHADsector[sec]= energyInEachChannel[2][sec]+energyInEachChannel[3][sec]+energyInEachChannel[4][sec]+
                          energyInEachChannel[5][sec]+energyInEachChannel[6][sec]+energyInEachChannel[7][sec]+
                          energyInEachChannel[8][sec]+energyInEachChannel[9][sec]+energyInEachChannel[10][sec]+
                          energyInEachChannel[11][sec]+energyInEachChannel[12][sec]+energyInEachChannel[13][sec];
      }

      meEsectorHAD1->Fill(energyHADsector[0]);
      meEsectorHAD2->Fill(energyHADsector[1]);
      meEsectorHAD3->Fill(energyHADsector[2]);
      meEsectorHAD4->Fill(energyHADsector[3]);
      meEsectorHAD5->Fill(energyHADsector[4]);
      meEsectorHAD6->Fill(energyHADsector[5]);
      meEsectorHAD7->Fill(energyHADsector[6]);
      meEsectorHAD8->Fill(energyHADsector[7]);
      meEsectorHAD9->Fill(energyHADsector[8]);
      meEsectorHAD10->Fill(energyHADsector[9]);
      meEsectorHAD11->Fill(energyHADsector[10]);
      meEsectorHAD12->Fill(energyHADsector[11]);
      meEsectorHAD13->Fill(energyHADsector[12]);
      meEsectorHAD14->Fill(energyHADsector[13]);
      meEsectorHAD15->Fill(energyHADsector[14]);
      meEsectorHAD16->Fill(energyHADsector[15]);

       //-----------------------------------------//
      //-- fill energy ratio EM/HAD in sectors  --//
      //------------------------------------------//
      if(energyHADsector[0]!=0) meEsectorEMvsHAD1->Fill((energyInEachChannel[0][0]+energyInEachChannel[1][0])/energyHADsector[0]);
      if(energyHADsector[1]!=0) meEsectorEMvsHAD2->Fill((energyInEachChannel[0][1]+energyInEachChannel[1][1])/energyHADsector[1]);
      if(energyHADsector[2]!=0) meEsectorEMvsHAD3->Fill((energyInEachChannel[0][2]+energyInEachChannel[1][2])/energyHADsector[2]);
      if(energyHADsector[3]!=0) meEsectorEMvsHAD4->Fill((energyInEachChannel[0][3]+energyInEachChannel[1][3])/energyHADsector[3]);
      if(energyHADsector[4]!=0) meEsectorEMvsHAD5->Fill((energyInEachChannel[0][4]+energyInEachChannel[1][4])/energyHADsector[4]);
      if(energyHADsector[5]!=0) meEsectorEMvsHAD6->Fill((energyInEachChannel[0][5]+energyInEachChannel[1][5])/energyHADsector[5]);
      if(energyHADsector[6]!=0) meEsectorEMvsHAD7->Fill((energyInEachChannel[0][6]+energyInEachChannel[1][6])/energyHADsector[6]);
      if(energyHADsector[7]!=0) meEsectorEMvsHAD8->Fill((energyInEachChannel[0][7]+energyInEachChannel[1][7])/energyHADsector[7]);
      if(energyHADsector[8]!=0) meEsectorEMvsHAD9->Fill((energyInEachChannel[0][8]+energyInEachChannel[1][8])/energyHADsector[8]);
      if(energyHADsector[9]!=0) meEsectorEMvsHAD10->Fill((energyInEachChannel[0][9]+energyInEachChannel[1][9])/energyHADsector[9]);
      if(energyHADsector[10]!=0) meEsectorEMvsHAD11->Fill((energyInEachChannel[0][10]+energyInEachChannel[1][10])/energyHADsector[10]);
      if(energyHADsector[11]!=0) meEsectorEMvsHAD12->Fill((energyInEachChannel[0][11]+energyInEachChannel[1][11])/energyHADsector[11]);
      if(energyHADsector[12]!=0) meEsectorEMvsHAD13->Fill((energyInEachChannel[0][12]+energyInEachChannel[1][12])/energyHADsector[12]);
      if(energyHADsector[13]!=0) meEsectorEMvsHAD14->Fill((energyInEachChannel[0][13]+energyInEachChannel[1][13])/energyHADsector[13]);
      if(energyHADsector[14]!=0) meEsectorEMvsHAD15->Fill((energyInEachChannel[0][14]+energyInEachChannel[1][14])/energyHADsector[14]);
      if(energyHADsector[15]!=0) meEsectorEMvsHAD16->Fill((energyInEachChannel[0][15]+energyInEachChannel[1][15])/energyHADsector[15]);


      //------------------------------------------//
      //--    fill the total EM energy per event --//
      //------------------------------------------//
 
      ////---- estimate the total energy deposited in the EM section in an event
     for(int mod=0; mod<2;mod++)
       for(int sec=0; sec<16;sec++)
	 EtotalEM += energyInEachChannel[mod][sec];
	 

       meEtotalEM->Fill(EtotalEM);

      //--------------------------------------------//
      //--    fill the total HAD energy per event --//
      //-------------------------------------------//
 
      ////---- estimate the total energy deposited in the EM section in an event
     for(int mod=2; mod<14;mod++)
       for(int sec=0; sec<16;sec++)
	 EtotalHAD += energyInEachChannel[mod][sec];
	 

       meEtotalHAD->Fill(EtotalHAD);

      //----------------------------------------------------//
      //--  fill the total energy ratio EM/HAD per event  --//
      //----------------------------------------------------//
 
      if(EtotalHAD!=0) meEtotalEMvsHAD->Fill(EtotalEM/EtotalHAD);

      //----------------------------------------------------//
      //--  fill the total energy in CASTOR per event     --//
      //----------------------------------------------------//
      for(int mod=0; mod<14;mod++)
        for(int sec=0; sec<16;sec++)
          EtotalCASTOR += energyInEachChannel[mod][sec];	 

      meEtotalCASTOR->Fill(EtotalCASTOR);
 

      //-------------------------------------------------------//
      //--  fill the total energy in CASTOR sectors per run --//
      //------------------------------------------------------//
  
     for(int sec=0; sec<16;sec++){

      ////--- sum over all modules
      energyTotalSector[sec] =  energyTotalChannel[0][sec]+energyTotalChannel[1][sec]+energyTotalChannel[2][sec]+
                                energyTotalChannel[3][sec]+energyTotalChannel[4][sec]+energyTotalChannel[5][sec]+
                                energyTotalChannel[6][sec]+energyTotalChannel[7][sec]+energyTotalChannel[8][sec]+
                                energyTotalChannel[9][sec]+energyTotalChannel[10][sec]+energyTotalChannel[11][sec]+
                                energyTotalChannel[12][sec]+energyTotalChannel[13][sec];
      }

      for(int sec=0; sec<16;sec++)
	meEtotalSector->Fill(sec,energyTotalSector[sec]);
 

	
  } //-- end of working with RecHits

    else { if(fVerbosity>0) std::cout<<"CastorHIMonitor::processEvent NO Castor RecHits !!!"<<std::endl; }



  //********************************************************//
  //************ working with Digis ***********************//
  //*******************************************************//



  if(castorDigis.size()>0) {

    ////--- leave it empty for the moment

  }





  else { if(fVerbosity>0) std::cout<<"CastorHIMonitor::processEvent NO Castor Digis !!!"<<std::endl; }


  if (showTiming) { 
      cpu_timer.stop(); std::cout << " TIMER::CastorRecHit -> " << cpu_timer.cpuTime() << std::endl; 
      cpu_timer.reset(); cpu_timer.start();  
    }
  
  ievt_++; 

  return;

}


