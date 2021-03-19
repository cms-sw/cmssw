#include "CommonTools/CandAlgos/interface/ModifyObjectValueBase.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

class LowPtGsfElectronFinalizer : public edm::stream::EDProducer<> {
public:
  explicit LowPtGsfElectronFinalizer(const edm::ParameterSet&);

  void produce(edm::Event&, const edm::EventSetup&) override;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  const edm::EDGetTokenT<reco::GsfElectronCollection> previousGsfElectrons_;
  std::unique_ptr<ModifyObjectValueBase> regression_;

  const edm::EDPutTokenT<reco::GsfElectronCollection> putToken_;
};

using edm::InputTag;
using reco::GsfElectronCollection;

LowPtGsfElectronFinalizer::LowPtGsfElectronFinalizer(const edm::ParameterSet& cfg)
    : previousGsfElectrons_{consumes<GsfElectronCollection>(cfg.getParameter<InputTag>("previousGsfElectronsTag"))},
      putToken_{produces<GsfElectronCollection>()} {
  auto const& iconf = cfg.getParameterSet("regressionConfig");
  auto const& mname = iconf.getParameter<std::string>("modifierName");
  auto cc = consumesCollector();
  regression_ = ModifyObjectValueFactory::get()->create(mname, iconf, cc);
}

void LowPtGsfElectronFinalizer::produce(edm::Event& event, const edm::EventSetup& setup) {
  // Setup regression for event
  regression_->setEvent(event);
  regression_->setEventContent(setup);

  // Create new modified electron collection
  reco::GsfElectronCollection outputElectrons;
  for (auto const& electron : event.get(previousGsfElectrons_)) {
    outputElectrons.emplace_back(electron);
    auto& newElectron = outputElectrons.back();
    regression_->modifyObject(newElectron);
  }

  // Emplace modified electrons to event
  event.emplace(putToken_, std::move(outputElectrons));
}

void LowPtGsfElectronFinalizer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("previousGsfElectronsTag", {});
  edm::ParameterSetDescription psd;
  psd.setUnknown();
  desc.add<edm::ParameterSetDescription>("regressionConfig", psd);
  descriptions.addWithDefaultLabel(desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(LowPtGsfElectronFinalizer);
