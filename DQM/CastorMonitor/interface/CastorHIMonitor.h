#ifndef DQM_CASTORMONITOR_CASTORHIMONITOR_H
#define DQM_CASTORMONITOR_CASTORHIMONITOR_H

#include "DQM/CastorMonitor/interface/CastorBaseMonitor.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
//#include "DataFormats/HcalRecHit/interface/CastorRecHit.h"

class CastorHIMonitor: public CastorBaseMonitor {
public:
  CastorHIMonitor(); 
  ~CastorHIMonitor(); 

  void setup(const edm::ParameterSet& ps, DQMStore* dbe);
  void processEvent(const CastorRecHitCollection& castorHits, const CastorDigiCollection& cast, const CastorDbService& cond);
  void reset();

private:  

  bool doPerChannel_;
  // float occThresh_;
  int ievt_;



 ////--- CASTOR channels
 int module;
 int sector;
 int zside;

 ////---- CASTOR RecHit energy and time 
 float energy;    
 float time;
      
 ////---- define the arrays
 float energyInEachChannel[14][16];
 float energyTotalChannel[14][16];
 float energyHADsector[16];
 float energySectors[16];
 float energyTotalSector[16];

 ////---- total energy deposited in the EM section 
 float EtotalEM;

 ////---- total energy deposited in the HAD section 
 float EtotalHAD;

 ////---- total energy deposited in the whole CASTOR  
 float EtotalCASTOR;



 
  ////---- define Monitoring elements

 MonitorElement* meEVT_;


////---- energy in sectors
 MonitorElement* meEsector1;  
 MonitorElement* meEsector2;  
 MonitorElement* meEsector3;  
 MonitorElement* meEsector4;  
 MonitorElement* meEsector5;  
 MonitorElement* meEsector6;  
 MonitorElement* meEsector7;  
 MonitorElement* meEsector8;  
 MonitorElement* meEsector9;  
 MonitorElement* meEsector10;  
 MonitorElement* meEsector11;  
 MonitorElement* meEsector12;  
 MonitorElement* meEsector13;  
 MonitorElement* meEsector14;  
 MonitorElement* meEsector15;  
 MonitorElement* meEsector16;  

////---- energy in modules
 MonitorElement* meEmodule1;  
 MonitorElement* meEmodule2;
 MonitorElement* meEmodule3;
 MonitorElement* meEmodule4;
 MonitorElement* meEmodule5;
 MonitorElement* meEmodule6;
 MonitorElement* meEmodule7;
 MonitorElement* meEmodule8;
 MonitorElement* meEmodule9;
 MonitorElement* meEmodule10;
 MonitorElement* meEmodule11;
 MonitorElement* meEmodule12;
 MonitorElement* meEmodule13;
 MonitorElement* meEmodule14;

////---- energy in EM sectors (sum in modules 1-2)
 MonitorElement* meEsectorEM1;
 MonitorElement* meEsectorEM2;
 MonitorElement* meEsectorEM3;
 MonitorElement* meEsectorEM4;
 MonitorElement* meEsectorEM5;
 MonitorElement* meEsectorEM6;
 MonitorElement* meEsectorEM7;
 MonitorElement* meEsectorEM8;
 MonitorElement* meEsectorEM9;
 MonitorElement* meEsectorEM10;
 MonitorElement* meEsectorEM11;
 MonitorElement* meEsectorEM12;
 MonitorElement* meEsectorEM13;
 MonitorElement* meEsectorEM14;
 MonitorElement* meEsectorEM15;
 MonitorElement* meEsectorEM16;

////---- energy in HAD sectors (sum in modules 3-14)
 MonitorElement* meEsectorHAD1;
 MonitorElement* meEsectorHAD2;
 MonitorElement* meEsectorHAD3;
 MonitorElement* meEsectorHAD4;
 MonitorElement* meEsectorHAD5;
 MonitorElement* meEsectorHAD6;
 MonitorElement* meEsectorHAD7;
 MonitorElement* meEsectorHAD8;
 MonitorElement* meEsectorHAD9;
 MonitorElement* meEsectorHAD10;
 MonitorElement* meEsectorHAD11;
 MonitorElement* meEsectorHAD12;
 MonitorElement* meEsectorHAD13;
 MonitorElement* meEsectorHAD14;
 MonitorElement* meEsectorHAD15;
 MonitorElement* meEsectorHAD16;


////---- energy ratio EM/HAD in sectors
 MonitorElement* meEsectorEMvsHAD1;
 MonitorElement* meEsectorEMvsHAD2; 
 MonitorElement* meEsectorEMvsHAD3; 
 MonitorElement* meEsectorEMvsHAD4; 
 MonitorElement* meEsectorEMvsHAD5; 
 MonitorElement* meEsectorEMvsHAD6; 
 MonitorElement* meEsectorEMvsHAD7; 
 MonitorElement* meEsectorEMvsHAD8; 
 MonitorElement* meEsectorEMvsHAD9; 
 MonitorElement* meEsectorEMvsHAD10; 
 MonitorElement* meEsectorEMvsHAD11; 
 MonitorElement* meEsectorEMvsHAD12; 
 MonitorElement* meEsectorEMvsHAD13; 
 MonitorElement* meEsectorEMvsHAD14; 
 MonitorElement* meEsectorEMvsHAD15; 
 MonitorElement* meEsectorEMvsHAD16; 

////--- total energy in CASTOR
 MonitorElement* meEtotalCASTOR;

////--- total energy in each CASTOR sector
 MonitorElement* meEtotalSector;

////--- total EM energy 
 MonitorElement* meEtotalEM;
////--- total HAD energy 
 MonitorElement* meEtotalHAD;
////--- total energy ratio EM/HAD per event 
 MonitorElement* meEtotalEMvsHAD;




////---- charge in sectors
 MonitorElement* meChargeSector1;  
 MonitorElement* meChargeSector2;  
 MonitorElement* meChargeSector3;  
 MonitorElement* meChargeSector4;  
 MonitorElement* meChargeSector5;  
 MonitorElement* meChargeSector6;  
 MonitorElement* meChargeSector7;  
 MonitorElement* meChargeSector8;  
 MonitorElement* meChargeSector9;  
 MonitorElement* meChargeSector10;  
 MonitorElement* meChargeSector11;  
 MonitorElement* meChargeSector12;  
 MonitorElement* meChargeSector13;  
 MonitorElement* meChargeSector14;  
 MonitorElement* meChargeSector15;  
 MonitorElement* meChargeSector16;  

////---- charge in modules
 MonitorElement* meChargeModule1;  
 MonitorElement* meChargeModule2;
 MonitorElement* meChargeModule3;
 MonitorElement* meChargeModule4;
 MonitorElement* meChargeModule5;
 MonitorElement* meChargeModule6;
 MonitorElement* meChargeModule7;
 MonitorElement* meChargeModule8;
 MonitorElement* meChargeModule9;
 MonitorElement* meChargeModule10;
 MonitorElement* meChargeModule11;
 MonitorElement* meChargeModule12;
 MonitorElement* meChargeModule13;
 MonitorElement* meChargeModule14;

////---- charge in EM sectors (sum in modules 1-2)
 MonitorElement* meChargeSectorEM1;
 MonitorElement* meChargeSectorEM2;
 MonitorElement* meChargeSectorEM3;
 MonitorElement* meChargeSectorEM4;
 MonitorElement* meChargeSectorEM5;
 MonitorElement* meChargeSectorEM6;
 MonitorElement* meChargeSectorEM7;
 MonitorElement* meChargeSectorEM8;
 MonitorElement* meChargeSectorEM9;
 MonitorElement* meChargeSectorEM10;
 MonitorElement* meChargeSectorEM11;
 MonitorElement* meChargeSectorEM12;
 MonitorElement* meChargeSectorEM13;
 MonitorElement* meChargeSectorEM14;
 MonitorElement* meChargeSectorEM15;
 MonitorElement* meChargeSectorEM16;

////---- charge in HAD sectors (sum in modules 3-14)
 MonitorElement* meChargeSectorHAD1;
 MonitorElement* meChargeSectorHAD2;
 MonitorElement* meChargeSectorHAD3;
 MonitorElement* meChargeSectorHAD4;
 MonitorElement* meChargeSectorHAD5;
 MonitorElement* meChargeSectorHAD6;
 MonitorElement* meChargeSectorHAD7;
 MonitorElement* meChargeSectorHAD8;
 MonitorElement* meChargeSectorHAD9;
 MonitorElement* meChargeSectorHAD10;
 MonitorElement* meChargeSectorHAD11;
 MonitorElement* meChargeSectorHAD12;
 MonitorElement* meChargeSectorHAD13;
 MonitorElement* meChargeSectorHAD14;
 MonitorElement* meChargeSectorHAD15;
 MonitorElement* meChargeSectorHAD16;


////---- charge ratio EM/HAD in sectors
 MonitorElement* meChargeSectorEMvsHAD1;
 MonitorElement* meChargeSectorEMvsHAD2; 
 MonitorElement* meChargeSectorEMvsHAD3; 
 MonitorElement* meChargeSectorEMvsHAD4; 
 MonitorElement* meChargeSectorEMvsHAD5; 
 MonitorElement* meChargeSectorEMvsHAD6; 
 MonitorElement* meChargeSectorEMvsHAD7; 
 MonitorElement* meChargeSectorEMvsHAD8; 
 MonitorElement* meChargeSectorEMvsHAD9; 
 MonitorElement* meChargeSectorEMvsHAD10; 
 MonitorElement* meChargeSectorEMvsHAD11; 
 MonitorElement* meChargeSectorEMvsHAD12; 
 MonitorElement* meChargeSectorEMvsHAD13; 
 MonitorElement* meChargeSectorEMvsHAD14; 
 MonitorElement* meChargeSectorEMvsHAD15; 
 MonitorElement* meChargeSectorEMvsHAD16; 

////--- total charge in CASTOR
 MonitorElement* meChargeTotalCASTOR;

////--- total charge in CASTOR sectors
 MonitorElement* meChargeTotalSectors;

////--- total EM charge 
 MonitorElement* meChargeTotalEM;
////--- total HAD charge 
 MonitorElement* meChargeTotalHAD;
////--- total charge ratio EM/HAD per event 
 MonitorElement* meChargeTotalEMvsHAD;

















};

#endif
