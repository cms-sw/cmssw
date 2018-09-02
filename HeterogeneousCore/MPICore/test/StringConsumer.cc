#include <iostream>
#include <string>

// user include files
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HLTrigger/HLTcore/interface/defaultModuleLabel.h"

class StringConsumer : public edm::global::EDAnalyzer<> {

  public:
    explicit StringConsumer(edm::ParameterSet const& config);
    ~StringConsumer() override = default;

    static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
    void analyze(edm::StreamID, edm::Event const& event, edm::EventSetup const& setup) const override;

  private:
    edm::EDGetTokenT<std::string> token_;
};

StringConsumer::StringConsumer(edm::ParameterSet const& config) :
  token_(consumes<std::string>(config.getParameter<edm::InputTag>("source")))
{
}

void StringConsumer::analyze(edm::StreamID sid, edm::Event const& event, edm::EventSetup const& setup) const {
  edm::Handle<std::string> handle;
  event.getByToken(token_, handle);
  std::cout << *handle << std::endl;
}

void StringConsumer::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("source", edm::InputTag());
  descriptions.add(defaultModuleLabel<StringConsumer>(), desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(StringConsumer);
