#include "DataFormats/EgammaCandidates/interface/GsfElectronCore.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronCoreFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/ParticleFlowReco/interface/GsfPFRecTrack.h"
#include "DataFormats/ParticleFlowReco/interface/GsfPFRecTrackFwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "RecoEgamma/EgammaElectronProducers/plugins/LowPtGsfElectronCoreProducer.h"

LowPtGsfElectronCoreProducer::LowPtGsfElectronCoreProducer(const edm::ParameterSet& config)
    : GsfElectronCoreBaseProducer(config) {
  superClusterRefs_ =
      consumes<edm::ValueMap<reco::SuperClusterRef> >(config.getParameter<edm::InputTag>("superClusters"));
}

void LowPtGsfElectronCoreProducer::produce(edm::Event& event, const edm::EventSetup& setup) {
  // Output collection
  auto electrons = std::make_unique<reco::GsfElectronCoreCollection>();

  // Init
  GsfElectronCoreBaseProducer::initEvent(event, setup);
  if (!useGsfPfRecTracks_) {
    edm::LogError("useGsfPfRecTracks_ is (redundantly) set to False!");
  }

  edm::Handle<edm::ValueMap<reco::SuperClusterRef> > superClusterRefs;
  event.getByToken(superClusterRefs_, superClusterRefs);

  // Create ElectronCore objects
  for (size_t ipfgsf = 0; ipfgsf < gsfPfRecTracksH_->size(); ++ipfgsf) {
    // Refs to GSF(PF) objects and SC
    reco::GsfPFRecTrackRef pfgsf(gsfPfRecTracksH_, ipfgsf);
    reco::GsfTrackRef gsf = pfgsf->gsfTrackRef();
    const reco::SuperClusterRef sc = (*superClusterRefs)[pfgsf];

    // Use GsfElectronCore(gsf) constructor and store object via emplace
    electrons->emplace_back(gsf);

    // Do not consider ECAL-driven objects
    if (electrons->back().ecalDrivenSeed()) {
      electrons->pop_back();
      continue;
    }

    // Add GSF(PF) track information
    GsfElectronCoreBaseProducer::fillElectronCore(&(electrons->back()));

    // Add super cluster information
    electrons->back().setSuperCluster(sc);
  }

  event.put(std::move(electrons));
}

//////////////////////////////////////////////////////////////////////////////////////////
//
void LowPtGsfElectronCoreProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  GsfElectronCoreBaseProducer::fillDescription(desc, "lowPtGsfElePfGsfTracks", "lowPtGsfEleGsfTracks");
  desc.add<edm::InputTag>("superClusters", edm::InputTag("lowPtGsfElectronSuperClusters"));
  descriptions.add("lowPtGsfElectronCores", desc);
}

//////////////////////////////////////////////////////////////////////////////////////////
//
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(LowPtGsfElectronCoreProducer);
