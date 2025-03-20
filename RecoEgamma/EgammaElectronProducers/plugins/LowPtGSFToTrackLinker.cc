#include <string>

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/Common/interface/Association.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/ParticleFlowReco/interface/PreId.h"
#include "DataFormats/ParticleFlowReco/interface/PreIdFwd.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/Framework/interface/MakerMacros.h"

class LowPtGSFToTrackLinker : public edm::global::EDProducer<> {
public:
  explicit LowPtGSFToTrackLinker(const edm::ParameterSet&);

  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  const edm::EDGetTokenT<reco::TrackCollection> tracks_;
  const edm::EDGetTokenT<std::vector<reco::PreId> > preid_;
  const edm::EDGetTokenT<std::vector<reco::GsfTrack> > gsftracks_;
};

LowPtGSFToTrackLinker::LowPtGSFToTrackLinker(const edm::ParameterSet& iConfig)
    : tracks_{consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("tracks"))},
      preid_{consumes<std::vector<reco::PreId> >(iConfig.getParameter<edm::InputTag>("gsfPreID"))},
      gsftracks_{consumes<std::vector<reco::GsfTrack> >(iConfig.getParameter<edm::InputTag>("gsfTracks"))} {
  produces<edm::Association<reco::TrackCollection> >();
}

void LowPtGSFToTrackLinker::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup&) const {
  auto gsftracks = iEvent.getHandle(gsftracks_);
  auto tracks = iEvent.getHandle(tracks_);
  auto preid = iEvent.getHandle(preid_);

  // collection sizes, for reference
  const size_t ngsf = gsftracks->size();

  //store mapping for association
  std::vector<int> gsf2track(ngsf, -1);

  //map Track --> GSF and fill GSF --> PackedCandidates and GSF --> Lost associations
  for (unsigned int igsf = 0; igsf < ngsf; ++igsf) {
    reco::GsfTrackRef gref(gsftracks, igsf);
    reco::TrackRef trk = gref->seedRef().castTo<reco::ElectronSeedRef>()->ctfTrack();

    if (trk.id() != tracks.id()) {
      throw cms::Exception(
          "WrongCollection",
          "The reco::Track collection used to match against the GSF Tracks was not used to produce such tracks");
    }

    size_t trkid = trk.index();
    gsf2track[igsf] = trkid;
  }

  // create output collections from the mappings
  auto assoc = std::make_unique<edm::Association<reco::TrackCollection> >(tracks);
  edm::Association<reco::TrackCollection>::Filler filler(*assoc);
  filler.insert(gsftracks, gsf2track.begin(), gsf2track.end());
  filler.fill();
  iEvent.put(std::move(assoc));
}

void LowPtGSFToTrackLinker::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("tracks", edm::InputTag("generalTracks"));
  desc.add<edm::InputTag>("gsfPreID", edm::InputTag("lowPtGsfElectronSeeds"));
  desc.add<edm::InputTag>("gsfTracks", edm::InputTag("lowPtGsfEleGsfTracks"));
  descriptions.add("lowPtGsfToTrackLinks", desc);
}

DEFINE_FWK_MODULE(LowPtGSFToTrackLinker);
