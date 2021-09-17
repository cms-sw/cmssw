#include <string>

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Common/interface/RefToPtr.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
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
#include "FWCore/Framework/interface/MakerMacros.h"

typedef edm::Ptr<pat::PackedCandidate> PackedCandidatePtr;
typedef std::vector<PackedCandidatePtr> PackedCandidatePtrCollection;

class LowPtGSFToPackedCandidateLinker : public edm::global::EDProducer<> {
public:
  explicit LowPtGSFToPackedCandidateLinker(const edm::ParameterSet&);
  ~LowPtGSFToPackedCandidateLinker() override;

  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  const edm::EDGetTokenT<reco::PFCandidateCollection> pfcands_;
  const edm::EDGetTokenT<pat::PackedCandidateCollection> packed_;
  const edm::EDGetTokenT<pat::PackedCandidateCollection> lost_tracks_;
  const edm::EDGetTokenT<reco::TrackCollection> tracks_;
  const edm::EDGetTokenT<edm::Association<pat::PackedCandidateCollection> > pf2packed_;
  const edm::EDGetTokenT<edm::Association<pat::PackedCandidateCollection> > lost2trk_;
  const edm::EDGetTokenT<edm::Association<reco::TrackCollection> > gsf2trk_;
  const edm::EDGetTokenT<std::vector<reco::GsfTrack> > gsftracks_;
  const edm::EDGetTokenT<std::vector<pat::Electron> > electrons_;
};

LowPtGSFToPackedCandidateLinker::LowPtGSFToPackedCandidateLinker(const edm::ParameterSet& iConfig)
    : pfcands_{consumes<reco::PFCandidateCollection>(iConfig.getParameter<edm::InputTag>("PFCandidates"))},
      packed_{consumes<pat::PackedCandidateCollection>(iConfig.getParameter<edm::InputTag>("packedCandidates"))},
      lost_tracks_{consumes<pat::PackedCandidateCollection>(iConfig.getParameter<edm::InputTag>("lostTracks"))},
      tracks_{consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("tracks"))},
      pf2packed_{consumes<edm::Association<pat::PackedCandidateCollection> >(
          iConfig.getParameter<edm::InputTag>("packedCandidates"))},
      lost2trk_{consumes<edm::Association<pat::PackedCandidateCollection> >(
          iConfig.getParameter<edm::InputTag>("lostTracks"))},
      gsf2trk_{consumes<edm::Association<reco::TrackCollection> >(iConfig.getParameter<edm::InputTag>("gsfToTrack"))},
      gsftracks_{consumes<std::vector<reco::GsfTrack> >(iConfig.getParameter<edm::InputTag>("gsfTracks"))},
      electrons_{consumes<std::vector<pat::Electron> >(iConfig.getParameter<edm::InputTag>("electrons"))} {
  produces<edm::Association<pat::PackedCandidateCollection> >("gsf2packed");
  produces<edm::Association<pat::PackedCandidateCollection> >("gsf2lost");
  produces<edm::ValueMap<PackedCandidatePtr> >("ele2packed");
  produces<edm::ValueMap<PackedCandidatePtr> >("ele2lost");
}

LowPtGSFToPackedCandidateLinker::~LowPtGSFToPackedCandidateLinker() {}

void LowPtGSFToPackedCandidateLinker::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  auto pfcands = iEvent.getHandle(pfcands_);
  auto packed = iEvent.getHandle(packed_);
  auto lost_tracks = iEvent.getHandle(lost_tracks_);
  auto pf2packed = iEvent.getHandle(pf2packed_);
  auto lost2trk_assoc = iEvent.getHandle(lost2trk_);
  auto gsftracks = iEvent.getHandle(gsftracks_);
  auto tracks = iEvent.getHandle(tracks_);
  auto gsf2trk = iEvent.getHandle(gsf2trk_);
  auto electrons = iEvent.getHandle(electrons_);

  // collection sizes, for reference
  const size_t npf = pfcands->size();
  const size_t npacked = packed->size();
  const size_t nlost = lost_tracks->size();
  const size_t ntracks = tracks->size();
  const size_t ngsf = gsftracks->size();
  const size_t nele = electrons->size();

  //store index mapping in vectors for easy and fast access
  std::vector<size_t> trk2packed(ntracks, npacked);
  std::vector<size_t> trk2lost(ntracks, nlost);

  //store auxiliary mappings for association
  std::vector<int> gsf2pack(ngsf, -1);
  std::vector<int> gsf2lost(ngsf, -1);
  PackedCandidatePtrCollection ele2packedptr(nele, PackedCandidatePtr(packed, -1));
  PackedCandidatePtrCollection ele2lostptr(nele, PackedCandidatePtr(lost_tracks, -1));

  //electrons will never store their track (they store the Gsf track)
  //map PackedPF <--> Track
  for (unsigned int icand = 0; icand < npf; ++icand) {
    edm::Ref<reco::PFCandidateCollection> pf_ref(pfcands, icand);
    const reco::PFCandidate& cand = pfcands->at(icand);
    auto packed_ref = (*pf2packed)[pf_ref];
    if (cand.charge() && packed_ref.isNonnull() && cand.trackRef().isNonnull() && cand.trackRef().id() == tracks.id()) {
      size_t trkid = cand.trackRef().index();
      trk2packed[trkid] = packed_ref.index();
    }
  }

  //map LostTrack <--> Track
  for (unsigned int itrk = 0; itrk < ntracks; ++itrk) {
    reco::TrackRef key(tracks, itrk);
    pat::PackedCandidateRef lostTrack = (*lost2trk_assoc)[key];
    if (lostTrack.isNonnull()) {
      trk2lost[itrk] = lostTrack.index();  // assumes that LostTracks are all made from the same track collection
    }
  }

  //map Track --> GSF and fill GSF --> PackedCandidates and GSF --> Lost associations
  for (unsigned int igsf = 0; igsf < ngsf; ++igsf) {
    reco::GsfTrackRef gsf_ref(gsftracks, igsf);
    reco::TrackRef trk_ref = (*gsf2trk)[gsf_ref];
    if (trk_ref.id() != tracks.id()) {
      throw cms::Exception(
          "WrongCollection",
          "The reco::Track collection used to match against the GSF Tracks was not used to produce such tracks");
    }
    size_t trkid = trk_ref.index();
    if (trk2packed[trkid] != npacked) {
      gsf2pack[igsf] = trk2packed[trkid];
    }
    if (trk2lost[trkid] != nlost) {
      gsf2lost[igsf] = trk2lost[trkid];
    }
  }

  //map Electron-->pat::PFCandidatePtr via Electron-->GsfTrack-->Track and Track-->pat::PFCandidatePtr
  for (unsigned int iele = 0; iele < nele; ++iele) {
    auto const& ele = (*electrons)[iele];
    reco::GsfTrackRef gsf_ref = ele.core()->gsfTrack();
    reco::TrackRef trk_ref = (*gsf2trk)[gsf_ref];
    if (trk_ref.id() != tracks.id()) {
      throw cms::Exception(
          "WrongCollection",
          "The reco::Track collection used to match against the GSF Tracks was not used to produce such tracks");
    }
    size_t trkid = trk_ref.index();
    auto packedIdx = trk2packed[trkid];
    if (packedIdx != npacked) {
      ele2packedptr[iele] = PackedCandidatePtr(packed, packedIdx);
    }
    auto lostIdx = trk2lost[trkid];
    if (lostIdx != nlost) {
      ele2lostptr[iele] = PackedCandidatePtr(lost_tracks, lostIdx);
    }
  }

  // create output collections from the mappings
  auto assoc_gsf2pack = std::make_unique<edm::Association<pat::PackedCandidateCollection> >(packed);
  edm::Association<pat::PackedCandidateCollection>::Filler gsf2pack_filler(*assoc_gsf2pack);
  gsf2pack_filler.insert(gsftracks, gsf2pack.begin(), gsf2pack.end());
  gsf2pack_filler.fill();
  iEvent.put(std::move(assoc_gsf2pack), "gsf2packed");

  auto assoc_gsf2lost = std::make_unique<edm::Association<pat::PackedCandidateCollection> >(lost_tracks);
  edm::Association<pat::PackedCandidateCollection>::Filler gsf2lost_filler(*assoc_gsf2lost);
  gsf2lost_filler.insert(gsftracks, gsf2lost.begin(), gsf2lost.end());
  gsf2lost_filler.fill();
  iEvent.put(std::move(assoc_gsf2lost), "gsf2lost");

  auto map_ele2packedptr = std::make_unique<edm::ValueMap<PackedCandidatePtr> >();
  edm::ValueMap<PackedCandidatePtr>::Filler ele2packedptr_filler(*map_ele2packedptr);
  ele2packedptr_filler.insert(electrons, ele2packedptr.begin(), ele2packedptr.end());
  ele2packedptr_filler.fill();
  iEvent.put(std::move(map_ele2packedptr), "ele2packed");

  auto map_ele2lostptr = std::make_unique<edm::ValueMap<PackedCandidatePtr> >();
  edm::ValueMap<PackedCandidatePtr>::Filler ele2lostptr_filler(*map_ele2lostptr);
  ele2lostptr_filler.insert(electrons, ele2lostptr.begin(), ele2lostptr.end());
  ele2lostptr_filler.fill();
  iEvent.put(std::move(map_ele2lostptr), "ele2lost");
}

void LowPtGSFToPackedCandidateLinker::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("PFCandidates", edm::InputTag("particleFlow"));
  desc.add<edm::InputTag>("packedCandidates", edm::InputTag("packedPFCandidates"));
  desc.add<edm::InputTag>("lostTracks", edm::InputTag("lostTracks"));
  desc.add<edm::InputTag>("tracks", edm::InputTag("generalTracks"));
  desc.add<edm::InputTag>("gsfToTrack", edm::InputTag("lowPtGsfToTrackLinks"));
  desc.add<edm::InputTag>("gsfTracks", edm::InputTag("lowPtGsfEleGsfTracks"));
  desc.add<edm::InputTag>("electrons", edm::InputTag("selectedPatLowPtElectrons"));
  descriptions.add("lowPtGsfLinksDefault", desc);
}

DEFINE_FWK_MODULE(LowPtGSFToPackedCandidateLinker);
