#ifndef HeterogeneousCore_CUDACore_interface_ConvertingESProducerT_h
#define HeterogeneousCore_CUDACore_interface_ConvertingESProducerT_h

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/eventsetuprecord_registration_macro.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/typelookup.h"

template <typename Record, typename Target, typename Source>
class ConvertingESProducerT : public edm::ESProducer {
public:
  explicit ConvertingESProducerT(edm::ParameterSet const& ps) {
    auto const& label = ps.getParameter<std::string>("label");
    auto const& name = ps.getParameter<std::string>("ComponentName");
    auto cc = setWhatProduced(this, name);
    token_ = cc.consumes(edm::ESInputTag{"", label});
  }

  std::unique_ptr<Target> produce(Record const& record) {
    // retrieve conditions in the old format and build a product in the new format
    return std::make_unique<Target>(record.get(token_));
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& confDesc) {
    edm::ParameterSetDescription desc;

    desc.add<std::string>("ComponentName", "");
    desc.add<std::string>("label", "")->setComment("ESProduct label");
    confDesc.addWithDefaultLabel(desc);
  }

private:
  edm::ESGetToken<Source, Record> token_;
};

#endif  // HeterogeneousCore_CUDACore_interface_ConvertingESProducerT_h
