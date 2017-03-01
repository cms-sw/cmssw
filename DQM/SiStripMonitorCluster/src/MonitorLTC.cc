// -*- C++ -*-
//
// Package:    SiStripMonitorCluster
// Class:      MonitorLTC
// 
//class MonitorLTC MonitorLTC.cc DQM/SiStripMonitorCluster/src/MonitorLTC.cc
#include <vector>

#include <numeric>
#include <iostream>



#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQM/SiStripMonitorCluster/interface/MonitorLTC.h"
#include "DQMServices/Core/interface/DQMStore.h"


MonitorLTC::MonitorLTC(const edm::ParameterSet& iConfig) // :
  //  ltcDigiCollectionTag_(iConfig.getParameter<edm::InputTag>("ltcDigiCollectionTag"))
{
  HLTDirectory="HLTResults";
  dqmStore_  = edm::Service<DQMStore>().operator->();
  conf_ = iConfig;

  ltcDigiCollectionTagToken_ = consumes<LTCDigiCollection>(conf_.getParameter<edm::InputTag>("ltcDigiCollectionTag") );
}

void MonitorLTC::bookHistograms(DQMStore::IBooker & ibooker, const edm::Run & run, const edm::EventSetup & es)
{
  ibooker.setCurrentFolder(HLTDirectory);
  // 0 DT
  // 1 CSC
  // 2 RBC1 (RPC techn. cosmic trigger for wheel +1, sector 10)
  // 3 RBC2 (RPC techn. cosmic trigger for wheel +2, sector 10)
  // 4 RPCTB (RPC Trigger Board trigger, covering both sectors 10 of both wheels, but with different geometrical acceptance ("pointing"))
  // 5 unused 
  // edm::CurrentProcessingContext const* current_processing_context = currentContext();
  // std::string const* the_label = moduleLabel();
  std::string the_label = conf_.getParameter<std::string>("@module_label");
  std::string ltctitle = the_label + "_LTCTriggerDecision";
  LTCTriggerDecision_all = ibooker.book1D(ltctitle, ltctitle, 8, -0.5, 7.5);
  LTCTriggerDecision_all->setBinLabel(1, "DT");
  LTCTriggerDecision_all->setBinLabel(2, "CSC");
  LTCTriggerDecision_all->setBinLabel(3, "RBC1");
  LTCTriggerDecision_all->setBinLabel(4, "RBC2");
  LTCTriggerDecision_all->setBinLabel(5, "RPCTB");  
}

void MonitorLTC::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  edm::Handle<LTCDigiCollection> ltcdigis; iEvent.getByToken(ltcDigiCollectionTagToken_, ltcdigis);
//  unsigned int ltc_run;
//  unsigned int ltc_event;
//  unsigned int ltc_triggerNumber;
//  unsigned int ltc_mask; // eight bits
//  unsigned long long int ltc_gpstime;
//  unsigned int ltc_orbit;
//  unsigned int ltc_bunch;
//  unsigned int ltc_inhibit;
  for( LTCDigiCollection::const_iterator ltcdigiItr = ltcdigis->begin() ; ltcdigiItr != ltcdigis->end() ; ++ltcdigiItr ) {
//    ltc_run = ltcdigiItr->runNumber();
//    ltc_event = ltcdigiItr->eventNumber();
//    ltc_triggerNumber = ltcdigiItr->eventID();
//    ltc_bunch = ltcdigiItr->bunchNumber();
//    ltc_orbit = ltcdigiItr->orbitNumber();
//    ltc_inhibit = ltcdigiItr->triggerInhibitNumber();
//    ltc_mask = (unsigned int)(ltcdigiItr->externTriggerMask());
//    ltc_gpstime = ltcdigiItr->bstGpsTime();
    for ( int ibit = 0; ibit < 7; ++ibit ) {
      if ( ltcdigiItr->HasTriggered(ibit) ) {
       LTCTriggerDecision_all->Fill(ibit,1.);
      }
    }
    //
//    std::cout << "XXX: "
//              << ltcdigiItr->runNumber() << " "
//              << ltcdigiItr->eventNumber() << " "
//              << ltcdigiItr->bunchNumber() << " "
//              << ltcdigiItr->orbitNumber() << " "
//              << "0x" << std::hex
//              << int(ltcdigiItr->externTriggerMask()) << " "
//              << std::dec
//              << ltcdigiItr->triggerInhibitNumber() << " "
//              << ltcdigiItr->bstGpsTime() << " "
//              << std::endl;
//    std::cout << (*ltcdigiItr) << std::endl;
  }

}
