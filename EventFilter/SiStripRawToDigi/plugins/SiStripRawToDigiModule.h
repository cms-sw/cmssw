#ifndef EventFilter_SiStripRawToDigi_SiStripRawToDigiModule_H
#define EventFilter_SiStripRawToDigi_SiStripRawToDigiModule_H

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include <string>
#include <cstdint>

namespace sistrip {
  class RawToDigiModule;
}
namespace sistrip {
  class RawToDigiUnpacker;
}
class SiStripFedCabling;

/**
   @file EventFilter/SiStripRawToDigi/interface/SiStripRawToDigiModule.h
   @class SiStripRawToDigiModule 
   
   @brief A plug-in module that takes a FEDRawDataCollection as input
   from the Event and creates EDProducts containing StripDigis.
*/

namespace sistrip {

  class dso_hidden RawToDigiModule final : public edm::stream::EDProducer<> {
  public:
    RawToDigiModule(const edm::ParameterSet&);
    ~RawToDigiModule() override;

    void beginRun(const edm::Run&, const edm::EventSetup&) override;
    void produce(edm::Event&, const edm::EventSetup&) override;
    void endStream() override;

  private:
    void updateCabling(const edm::EventSetup&);

    RawToDigiUnpacker* rawToDigi_;
    edm::EDGetTokenT<FEDRawDataCollection> token_;
    const SiStripFedCabling* cabling_;
    uint32_t cacheId_;
    bool extractCm_;
    bool doFullCorruptBufferChecks_;

    //March 2012: add flag for disabling APVe check in configuration
    bool doAPVEmulatorCheck_;
  };

}  // namespace sistrip

#endif  // EventFilter_SiStripRawToDigi_SiStripRawToDigiModule_H
