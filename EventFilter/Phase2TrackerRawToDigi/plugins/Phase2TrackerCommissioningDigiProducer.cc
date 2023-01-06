#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/Phase2TrackerDigi/interface/Phase2TrackerCommissioningDigi.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2TrackerFEDBuffer.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/Phase2TrackerFEDHeader.h"
#include "EventFilter/Phase2TrackerRawToDigi/interface/utils.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace Phase2Tracker {

  class Phase2TrackerCommissioningDigiProducer : public edm::global::EDProducer<> {
  public:
    /// constructor
    Phase2TrackerCommissioningDigiProducer(const edm::ParameterSet& pset);
    /// default constructor
    ~Phase2TrackerCommissioningDigiProducer() override = default;
    void produce(edm::StreamID, edm::Event& ev, const edm::EventSetup& es) const override;

  private:
    edm::EDGetTokenT<FEDRawDataCollection> token_;
  };
}  // namespace Phase2Tracker

#include "FWCore/Framework/interface/MakerMacros.h"
typedef Phase2Tracker::Phase2TrackerCommissioningDigiProducer Phase2TrackerCommissioningDigiProducer;
DEFINE_FWK_MODULE(Phase2TrackerCommissioningDigiProducer);

using namespace std;

Phase2Tracker::Phase2TrackerCommissioningDigiProducer::Phase2TrackerCommissioningDigiProducer(
    const edm::ParameterSet& pset) {
  produces<edm::DetSet<Phase2TrackerCommissioningDigi>>("ConditionData");
  token_ = consumes<FEDRawDataCollection>(pset.getParameter<edm::InputTag>("ProductLabel"));
}

void Phase2Tracker::Phase2TrackerCommissioningDigiProducer::produce(edm::StreamID,
                                                                    edm::Event& event,
                                                                    const edm::EventSetup& es) const {
  // Retrieve FEDRawData collection
  edm::Handle<FEDRawDataCollection> buffers;
  event.getByToken(token_, buffers);

  // Analyze strip tracker FED buffers in data
  size_t fedIndex;
  for (fedIndex = 0; fedIndex <= Phase2Tracker::CMS_FED_ID_MAX; ++fedIndex) {
    const FEDRawData& fed = buffers->FEDData(fedIndex);
    if (fed.size() != 0 && fedIndex >= Phase2Tracker::FED_ID_MIN && fedIndex <= Phase2Tracker::FED_ID_MAX) {
      // construct buffer
      Phase2Tracker::Phase2TrackerFEDBuffer* buffer = nullptr;
      buffer = new Phase2Tracker::Phase2TrackerFEDBuffer(fed.data(), fed.size());

      // fetch condition data
      std::map<uint32_t, uint32_t> cond_data = buffer->conditionData();
      delete buffer;

      // print cond data for debug
      LogTrace("Phase2TrackerCommissioningDigiProducer") << "--- Condition data debug ---" << std::endl;
      std::map<uint32_t, uint32_t>::const_iterator it;
      for (it = cond_data.begin(); it != cond_data.end(); it++) {
        LogTrace("Phase2TrackerCommissioningDigiProducer")
            << std::hex << "key: " << it->first << std::hex << " value: " << it->second << " (hex) " << std::dec
            << it->second << " (dec) " << std::endl;
      }
      LogTrace("Phase2TrackerCommissioningDigiProducer") << "----------------------------" << std::endl;
      // store it into digis
      edm::DetSet<Phase2TrackerCommissioningDigi>* cond_data_digi =
          new edm::DetSet<Phase2TrackerCommissioningDigi>(fedIndex);
      for (it = cond_data.begin(); it != cond_data.end(); it++) {
        cond_data_digi->push_back(Phase2TrackerCommissioningDigi(it->first, it->second));
      }
      std::unique_ptr<edm::DetSet<Phase2TrackerCommissioningDigi>> cdd(cond_data_digi);
      event.put(std::move(cdd), "ConditionData");
    }
  }
}
