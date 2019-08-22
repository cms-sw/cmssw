#ifndef EventFilter_Phase2TrackerRawToDigi_Phase2TrackerCommissioningDigiProducer_H
#define EventFilter_Phase2TrackerRawToDigi_Phase2TrackerCommissioningDigiProducer_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

namespace Phase2Tracker {

  class Phase2TrackerCommissioningDigiProducer : public edm::EDProducer {
  public:
    /// constructor
    Phase2TrackerCommissioningDigiProducer(const edm::ParameterSet& pset);
    /// default constructor
    ~Phase2TrackerCommissioningDigiProducer() override;
    void produce(edm::Event& event, const edm::EventSetup& es) override;

  private:
    edm::EDGetTokenT<FEDRawDataCollection> token_;
  };
}  // namespace Phase2Tracker
#endif  // EventFilter_Phase2TrackerRawToDigi_Phase2TrackerCommissioningDigiProducer_H
