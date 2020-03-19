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
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "CondFormats/SiStripObjects/interface/Phase2TrackerCabling.h"
#include "DataFormats/Phase2TrackerDigi/interface/Phase2TrackerDigi.h"
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace Phase2Tracker {

  class Phase2TrackerDigiProducer : public edm::EDProducer {
  public:
    /// constructor
    Phase2TrackerDigiProducer(const edm::ParameterSet& pset);
    /// default constructor
    ~Phase2TrackerDigiProducer() override;
    void beginJob() override;
    void beginRun(edm::Run const&, edm::EventSetup const&) override;
    void produce(edm::Event&, const edm::EventSetup&) override;
    void endJob() override;

  private:
    unsigned int runNumber_;
    edm::EDGetTokenT<FEDRawDataCollection> token_;
    const Phase2TrackerCabling* cabling_;
    uint32_t cacheId_;
    DetIdCollection detids_;
    class Registry {
    public:
      /// constructor
      Registry(uint32_t aDetid, uint16_t firstStrip, size_t indexInVector, uint16_t numberOfDigis)
          : detid(aDetid), first(firstStrip), index(indexInVector), length(numberOfDigis) {}
      /// < operator to sort registries
      bool operator<(const Registry& other) const {
        return (detid != other.detid ? detid < other.detid : first < other.first);
      }
      /// public data members
      uint32_t detid;
      uint16_t first;
      size_t index;
      uint16_t length;
    };
    std::vector<Registry> proc_work_registry_;
    std::vector<Phase2TrackerDigi> proc_work_digis_;
  };
}  // namespace Phase2Tracker
#endif  // EventFilter_Phase2TrackerRawToDigi_Phase2TrackerDigiProducer_H
