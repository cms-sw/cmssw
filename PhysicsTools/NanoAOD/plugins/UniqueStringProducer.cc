#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/NanoAOD/interface/UniqueString.h"

#include <vector>

class UniqueStringProducer : public edm::global::EDProducer<edm::BeginRunProducer> {
public:
  UniqueStringProducer(edm::ParameterSet const& iConfig) {
    const edm::ParameterSet& strings = iConfig.getParameter<edm::ParameterSet>("strings");
    for (const std::string& vname : strings.getParameterNamesForType<std::string>()) {
      strings_.emplace_back(vname, strings.getParameter<std::string>(vname));
      produces<nanoaod::UniqueString, edm::InRun>(vname);
    }
  }

  ~UniqueStringProducer() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    edm::ParameterSetDescription strings;
    strings.setComment("a parameter set definining the a pair of strings to be added in run flat table");
    strings.addNode(edm::ParameterWildcard<std::string>("*", edm::RequireZeroOrMore, true));
    desc.add<edm::ParameterSetDescription>("strings", strings);

    descriptions.addWithDefaultLabel(desc);
  }
  void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override {}  // do nothing

  void globalBeginRunProduce(edm::Run& iRun, edm::EventSetup const&) const override {
    for (const auto& pair : strings_) {
      iRun.put(std::make_unique<nanoaod::UniqueString>(pair.second), pair.first);
    }
  }

protected:
  std::vector<std::pair<std::string, std::string>> strings_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(UniqueStringProducer);
