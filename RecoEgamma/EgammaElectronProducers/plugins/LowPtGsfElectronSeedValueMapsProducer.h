#ifndef RecoEgamma_EgammaElectronProducers_LowPtGsfElectronSeedValueMapsProducer_h
#define RecoEgamma_EgammaElectronProducers_LowPtGsfElectronSeedValueMapsProducer_h

#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PreIdFwd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include <vector>
#include <string>

class LowPtGsfElectronSeedValueMapsProducer : public edm::stream::EDProducer<> {
public:
  explicit LowPtGsfElectronSeedValueMapsProducer(const edm::ParameterSet&);

  ~LowPtGsfElectronSeedValueMapsProducer() override;

  void produce(edm::Event&, const edm::EventSetup&) override;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  const edm::EDGetTokenT<reco::GsfTrackCollection> gsfTracks_;
  const edm::EDGetTokenT<edm::ValueMap<reco::PreIdRef> > preIdsValueMap_;
  const std::vector<std::string> names_;
};

#endif  // RecoEgamma_EgammaElectronProducers_LowPtGsfElectronSeedValueMapsProducer_h
