#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackExtra.h"
#include "DataFormats/ParticleFlowReco/interface/PreId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
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

  void produce(edm::Event&, const edm::EventSetup&) override;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  const edm::EDGetTokenT<reco::GsfTrackCollection> gsfTracks_;
  const edm::EDGetTokenT<edm::ValueMap<reco::PreIdRef> > preIdsValueMap_;
  const std::vector<std::string> names_;
};

////////////////////////////////////////////////////////////////////////////////
//
LowPtGsfElectronSeedValueMapsProducer::LowPtGsfElectronSeedValueMapsProducer(const edm::ParameterSet& conf)
    : gsfTracks_(consumes<reco::GsfTrackCollection>(conf.getParameter<edm::InputTag>("gsfTracks"))),
      preIdsValueMap_(consumes<edm::ValueMap<reco::PreIdRef> >(conf.getParameter<edm::InputTag>("preIdsValueMap"))),
      names_(conf.getParameter<std::vector<std::string> >("ModelNames")) {
  for (const auto& name : names_) {
    produces<edm::ValueMap<float> >(name);
  }
}

////////////////////////////////////////////////////////////////////////////////
//
void LowPtGsfElectronSeedValueMapsProducer::produce(edm::Event& event, const edm::EventSetup&) {
  // Retrieve GsfTracks from Event
  edm::Handle<reco::GsfTrackCollection> gsfTracks;
  event.getByToken(gsfTracks_, gsfTracks);
  if (!gsfTracks.isValid()) {
    edm::LogError("Problem with gsfTracks handle");
  }

  // Retrieve PreIds from Event
  edm::Handle<edm::ValueMap<reco::PreIdRef> > preIdsValueMap;
  event.getByToken(preIdsValueMap_, preIdsValueMap);
  if (!preIdsValueMap.isValid()) {
    edm::LogError("Problem with preIdsValueMap handle");
  }

  // Iterate through GsfTracks, extract BDT output, and store result in ValueMap for each model
  std::vector<std::vector<float> > output;
  for (unsigned int iname = 0; iname < names_.size(); ++iname) {
    output.push_back(std::vector<float>(gsfTracks->size(), -999.));
  }
  for (unsigned int igsf = 0; igsf < gsfTracks->size(); igsf++) {
    reco::GsfTrackRef gsf(gsfTracks, igsf);
    if (gsf.isNonnull() && gsf->extra().isNonnull() && gsf->extra()->seedRef().isNonnull()) {
      reco::ElectronSeedRef seed = gsf->extra()->seedRef().castTo<reco::ElectronSeedRef>();
      if (seed.isNonnull() && seed->ctfTrack().isNonnull()) {
        const reco::PreIdRef preid = (*preIdsValueMap)[seed->ctfTrack()];
        if (preid.isNonnull()) {
          for (unsigned int iname = 0; iname < names_.size(); ++iname) {
            output[iname][igsf] = preid->mva(iname);
          }
        }
      }
    }
  }

  // Create and put ValueMap in Event
  for (unsigned int iname = 0; iname < names_.size(); ++iname) {
    auto ptr = std::make_unique<edm::ValueMap<float> >(edm::ValueMap<float>());
    edm::ValueMap<float>::Filler filler(*ptr);
    filler.insert(gsfTracks, output[iname].begin(), output[iname].end());
    filler.fill();
    event.put(std::move(ptr), names_[iname]);
  }
}

//////////////////////////////////////////////////////////////////////////////////////////
//
void LowPtGsfElectronSeedValueMapsProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("gsfTracks", edm::InputTag("lowPtGsfEleGsfTracks"));
  desc.add<edm::InputTag>("preIdsValueMap", edm::InputTag("lowPtGsfElectronSeeds"));
  desc.add<std::vector<std::string> >("ModelNames", {"unbiased", "ptbiased"});
  descriptions.add("lowPtGsfElectronSeedValueMaps", desc);
}

//////////////////////////////////////////////////////////////////////////////////////////
//
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(LowPtGsfElectronSeedValueMapsProducer);
