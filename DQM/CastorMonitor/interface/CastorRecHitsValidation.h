#ifndef DQM_CASTORMONITOR_CASTORRECHITSVALIDATION_H
#define DQM_CASTORMONITOR_CASTORRECHITSVALIDATION_H


#include "DQM/CastorMonitor/interface/CastorBaseMonitor.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/HcalDigi/interface/CastorDataFrame.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <map>


class CastorRecHitsValidation: public CastorBaseMonitor{

  typedef std::map<uint32_t,float,std::less<uint32_t> >  MapType;

public:

 CastorRecHitsValidation(); 
  ~CastorRecHitsValidation(); 

  void setup(const edm::ParameterSet& ps, DQMStore* dbe);
  void processEvent(const CastorRecHitCollection& castorHits);
  void reset();


private:


int ievt_;
std::string histo;

MonitorElement* meCastorRecHitsEnergy_;
MonitorElement* meCastorRecHitsTotalEnergy_;
MonitorElement* meCastorRecHitsModule_;
MonitorElement* meCastorRecHitsSector_;
MonitorElement* meCastorRecHitsOccupancy_;

MonitorElement* meCastorRecHitsEmodule_;
MonitorElement* meCastorRecHitsEsector_;
MonitorElement* meCastorRecHitsX_;
MonitorElement* meCastorRecHitsY_;
MonitorElement* meCastorRecHitsZ_;
MonitorElement* meCastorRecHitsXY_;
MonitorElement* meCastorRecHitsZside_;

MonitorElement* meCastorRecHitsEta_;
MonitorElement* meCastorRecHitsPhi_;
MonitorElement* meCastorRecHitsGlobalX_;

//test
MonitorElement* meCastorRecHitsXYEMsector1_;
MonitorElement* meCastorRecHitsXYsector1_;

MonitorElement* meCastorSimHitsEnergy_;

};



#endif
