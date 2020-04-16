
#ifndef GsfElectronBaseProducer_h
#define GsfElectronBaseProducer_h

#include "RecoEgamma/EgammaElectronAlgos/interface/GsfElectronAlgo.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

namespace reco {
  class GsfElectron;
}

namespace edm {
  class ParameterSet;
  class ConfigurationDescriptions;
}  // namespace edm

#include "RecoEgamma/EgammaElectronAlgos/interface/GsfElectronAlgo.h"
#include "DataFormats/Common/interface/Handle.h"

class GsfElectronBaseProducer : public edm::stream::EDProducer<edm::GlobalCache<GsfElectronAlgo::HeavyObjectCache>> {
public:
  static void fillDescriptions(edm::ConfigurationDescriptions&);

  explicit GsfElectronBaseProducer(const edm::ParameterSet&, const GsfElectronAlgo::HeavyObjectCache*);
  ~GsfElectronBaseProducer() override;

  static std::unique_ptr<GsfElectronAlgo::HeavyObjectCache> initializeGlobalCache(const edm::ParameterSet& conf) {
    return std::make_unique<GsfElectronAlgo::HeavyObjectCache>(conf);
  }

  static void globalEndJob(GsfElectronAlgo::HeavyObjectCache const*) {}

  // ------------ method called to produce the data  ------------
  void produce(edm::Event& event, const edm::EventSetup& setup) override {
    auto electrons = algo_->completeElectrons(event, setup, globalCache());
    fillEvent(electrons, event);
  }

protected:
  std::unique_ptr<GsfElectronAlgo> algo_;

  void beginEvent(edm::Event&, const edm::EventSetup&);
  const edm::OrphanHandle<reco::GsfElectronCollection> fillEvent(reco::GsfElectronCollection& electrons,
                                                                 edm::Event& event);

  // configurables
  GsfElectronAlgo::Tokens inputCfg_;
  GsfElectronAlgo::StrategyConfiguration strategyCfg_;
  const GsfElectronAlgo::CutsConfiguration cutsCfg_;
  ElectronHcalHelper::Configuration hcalCfg_;

private:
  bool isPreselected(reco::GsfElectron const& ele) const;
  void setAmbiguityData(reco::GsfElectronCollection& electrons,
                        edm::Event const& event,
                        bool ignoreNotPreselected = true) const;

  // check expected configuration of previous modules
  bool ecalSeedingParametersChecked_;
  void checkEcalSeedingParameters(edm::ParameterSet const&);

  const edm::EDPutTokenT<reco::GsfElectronCollection> electronPutToken_;

  const edm::EDGetTokenT<reco::GsfPFRecTrackCollection> gsfPfRecTracksTag_;

  const bool gedElectronMode_;
  const bool useGsfPfRecTracks_;
};

#endif
