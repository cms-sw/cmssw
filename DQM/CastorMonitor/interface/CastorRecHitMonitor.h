#ifndef DQM_CASTORMONITOR_CASTORRECHITMONITOR_H
#define DQM_CASTORMONITOR_CASTORRECHITMONITOR_H

#include "DQM/CastorMonitor/interface/CastorBaseMonitor.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
//#include "DataFormats/HcalRecHit/interface/CastorRecHit.h"

class CastorRecHitMonitor: public CastorBaseMonitor {
public:
  CastorRecHitMonitor(); 
  ~CastorRecHitMonitor(); 

  void setup(const edm::ParameterSet& ps, DQMStore* dbe);
  void processEvent(const CastorRecHitCollection& castorHits);
  void reset();

private:  

  bool doPerChannel_;
  // float occThresh_;
  int ievt_;

 ////--- CASTOR channels
 int module;
 int sector;
 int zside;
 float channel;

 ////---- energy and time for every hit:
 float energy ;
 float time;
 float energyInEachChannel[14][16];
 float totEnergy;


 float  allEnergyModule[14];
 float  allEnergySector[16]; 

  ////---- define Monitoring elements
  struct{
    MonitorElement* meRECHIT_E_all     ;   //-- energy of all hits 
    MonitorElement* meRECHIT_T_all     ;   //-- time of all hits
    MonitorElement* meRECHIT_MAP_CHAN_E;   //-- energy vs channel plot
    MonitorElement* meRECHIT_E_modules;    //-- energy in modules
    MonitorElement* meRECHIT_E_sectors;    //-- energy in sectors 
    MonitorElement* meRECHIT_E_relative_modules;    //-- relative energy in modules
    MonitorElement* meRECHIT_E_relative_sectors;    //-- relative energy in sectors 
    MonitorElement* meRECHIT_N_modules;    //-- number of rechits in modules
    MonitorElement* meRECHIT_N_sectors;    //-- number of rechits in sectors 
    MonitorElement* meCastorRecHitsOccupancy; //-- occupancy plot
    MonitorElement* meRECHIT_N_event; //-- number of rechits per event
    std::map<HcalCastorDetId, MonitorElement*> meRECHIT_E, meRECHIT_T;  //-- complicated per-channel histogram setup
  } castorHists ;

  MonitorElement* meEVT_;

};

#endif
