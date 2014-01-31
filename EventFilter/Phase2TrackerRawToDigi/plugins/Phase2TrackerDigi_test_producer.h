#ifndef EventFilter_Phase2TrackerRawToDigi_Phase2TrackerDigi_test_producer_H
#define EventFilter_Phase2TrackerRawToDigi_Phase2TrackerDigi_test_producer_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/DetId/interface/DetIdCollection.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2TrackerFEDBuffer.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "DataFormats/Phase2TrackerDigi/interface/Phase2TrackerDigi.h"
#include "boost/cstdint.hpp"
#include <iostream>
#include <string>
#include <vector>

namespace sistrip {

  class Phase2TrackerDigi_test_producer : public edm::EDProducer
  {
  public:
    /// constructor
    Phase2TrackerDigi_test_producer( const edm::ParameterSet& pset );
    /// default constructor
    ~Phase2TrackerDigi_test_producer();
    void beginJob( const edm::EventSetup & es);
    void beginRun( edm::Run & run, const edm::EventSetup & es);
    void produce( edm::Event& event, const edm::EventSetup& es );
    void endJob();
    
  private:
    unsigned int runNumber_;
    edm::InputTag productLabel_;
    const SiStripFedCabling * cabling_;
    uint32_t cacheId_;
    DetIdCollection detids_;

    std::vector<Registry> proc_work_registry_;
    std::vector<Phase2TrackerDigi> proc_work_digis_;
  };
}
#endif // EventFilter_Phase2TrackerRawToDigi_Phase2TrackerDigi_test_producer_H
