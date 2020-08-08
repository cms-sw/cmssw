#ifndef EventFilter_EcalRawToDigi_plugins_EcalRawESProducerGPU_h
#define EventFilter_EcalRawToDigi_plugins_EcalRawESProducerGPU_h

#include <iostream>

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/eventsetuprecord_registration_macro.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/typelookup.h"

template <typename Target, typename Source, typename Record>
class EcalRawESProducerGPU : public edm::ESProducer {
public:
  explicit EcalRawESProducerGPU(edm::ParameterSet const& ps) {
    auto const label = ps.getParameter<std::string>("label");
    auto name = ps.getParameter<std::string>("ComponentName");
    auto cc = setWhatProduced(this, name);
    cc.setConsumes(token_, edm::ESInputTag{"", label});
  }

  std::unique_ptr<Target> produce(Record const& record) {
    // retrieve conditions in old format
    auto sourceProduct = record.getTransientHandle(token_);

    return std::make_unique<Target>(*sourceProduct);
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& confDesc) {
    edm::ParameterSetDescription desc;

    std::string label = Target::name() + "ESProducer";
    desc.add<std::string>("ComponentName", "");
    desc.add<std::string>("label", "")->setComment("Product Label");
    confDesc.add(label, desc);
  }

private:
  edm::ESGetToken<Source, Record> token_;
};

#endif  // EventFilter_EcalRawToDigi_plugins_EcalRawESProducerGPU_h
