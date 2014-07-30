#ifndef EventFilter_Phase2TrackerRawToDigi_Phase2TrackerDigiProducer_H
#define EventFilter_Phase2TrackerRawToDigi_Phase2TrackerDigiProducer_H

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
#include "CondFormats/SiStripObjects/interface/Phase2TrackerCabling.h"
#include "DataFormats/Phase2TrackerDigi/interface/Phase2TrackerDigi.h"
#include "boost/cstdint.hpp"
#include <iostream>
#include <string>
#include <vector>

namespace sistrip {

  class Phase2TrackerDigiProducer : public edm::EDProducer
  {
  public:
    /// constructor
    Phase2TrackerDigiProducer( const edm::ParameterSet& pset );
    /// default constructor
    ~Phase2TrackerDigiProducer();
    virtual void beginJob() override;
    virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
    virtual void produce(edm::Event&, const edm::EventSetup&) override;
    virtual void endJob() override;

    
  private:
    unsigned int runNumber_;
    edm::InputTag productLabel_;
    const Phase2TrackerCabling * cabling_;
    uint32_t cacheId_;
    DetIdCollection detids_;

    std::vector<Registry> proc_work_registry_;
    std::vector<Phase2TrackerDigi> proc_work_digis_;
  };
}
#endif // EventFilter_Phase2TrackerRawToDigi_Phase2TrackerDigiProducer_H
