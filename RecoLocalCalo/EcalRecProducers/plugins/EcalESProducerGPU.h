#ifndef RecoLocalCalo_EcalRecProducers_src_EcalESProducerGPU_h
#define RecoLocalCalo_EcalRecProducers_src_EcalESProducerGPU_h

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/typelookup.h"
#include "FWCore/Framework/interface/eventsetuprecord_registration_macro.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <iostream>

template <typename Target, typename Source, typename Record>
class EcalESProducerGPU : public edm::ESProducer {
public:
  explicit EcalESProducerGPU(edm::ParameterSet const& ps) {
    auto const label = ps.getParameter<std::string>("label");
    std::string name = ps.getParameter<std::string>("ComponentName");
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

#endif
