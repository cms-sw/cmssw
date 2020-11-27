#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

namespace {
  bool passesQuality(const reco::Track& trk, const std::vector<reco::TrackBase::TrackQuality>& allowedQuals) {
    for (const auto& qual : allowedQuals) {
      if (trk.quality(qual))
        return true;
    }
    return false;
  }
}  // namespace

namespace pat {
  class PATLostTracks : public edm::global::EDProducer<> {
  public:
    explicit PATLostTracks(const edm::ParameterSet&);
    ~PATLostTracks() override;

    void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  private:
    enum class TrkStatus { NOTUSED = 0, PFCAND, PFCANDNOTRKPROPS, PFELECTRON, PFPOSITRON, VTX };
    bool passTrkCuts(const reco::Track& tr) const;
    void addPackedCandidate(std::vector<pat::PackedCandidate>& cands,
                            const reco::TrackRef& trk,
                            const reco::VertexRef& pvSlimmed,
                            const reco::VertexRefProd& pvSlimmedColl,
                            const TrkStatus& trkStatus,
                            const pat::PackedCandidate::PVAssociationQuality& pvAssocQuality,
                            edm::Handle<reco::MuonCollection> muons) const;
    std::pair<int, pat::PackedCandidate::PVAssociationQuality> associateTrkToVtx(const reco::VertexCollection& vertices,
                                                                                 const reco::TrackRef& trk) const;

  private:
    const edm::EDGetTokenT<reco::PFCandidateCollection> cands_;
    const edm::EDGetTokenT<edm::Association<pat::PackedCandidateCollection>> map_;
    const edm::EDGetTokenT<reco::TrackCollection> tracks_;
    const edm::EDGetTokenT<reco::VertexCollection> vertices_;
    const edm::EDGetTokenT<reco::VertexCompositeCandidateCollection> kshorts_;
    const edm::EDGetTokenT<reco::VertexCompositeCandidateCollection> lambdas_;
    const edm::EDGetTokenT<reco::VertexCollection> pv_;
    const edm::EDGetTokenT<reco::VertexCollection> pvOrigs_;
    const double minPt_;
    const double minHits_;
    const double minPixelHits_;
    const double minPtToStoreProps_;
    const int covarianceVersion_;
    const std::vector<int> covariancePackingSchemas_;
    std::vector<reco::TrackBase::TrackQuality> qualsToAutoAccept_;
    const edm::EDGetTokenT<reco::MuonCollection> muons_;
    StringCutObjectSelector<reco::Track, false> passThroughCut_;
    const double maxDzForPrimaryAssignment_;
    const double maxDzSigForPrimaryAssignment_;
    const double maxDzErrorForPrimaryAssignment_;
    const double maxDxyForNotReconstructedPrimary_;
    const double maxDxySigForNotReconstructedPrimary_;
    const bool useLegacySetup_;
  };
}  // namespace pat

pat::PATLostTracks::PATLostTracks(const edm::ParameterSet& iConfig)
    : cands_(consumes<reco::PFCandidateCollection>(iConfig.getParameter<edm::InputTag>("inputCandidates"))),
      map_(consumes<edm::Association<pat::PackedCandidateCollection>>(
          iConfig.getParameter<edm::InputTag>("packedPFCandidates"))),
      tracks_(consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("inputTracks"))),
      vertices_(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("secondaryVertices"))),
      kshorts_(consumes<reco::VertexCompositeCandidateCollection>(iConfig.getParameter<edm::InputTag>("kshorts"))),
      lambdas_(consumes<reco::VertexCompositeCandidateCollection>(iConfig.getParameter<edm::InputTag>("lambdas"))),
      pv_(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("primaryVertices"))),
      pvOrigs_(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("originalVertices"))),
      minPt_(iConfig.getParameter<double>("minPt")),
      minHits_(iConfig.getParameter<uint32_t>("minHits")),
      minPixelHits_(iConfig.getParameter<uint32_t>("minPixelHits")),
      minPtToStoreProps_(iConfig.getParameter<double>("minPtToStoreProps")),
      covarianceVersion_(iConfig.getParameter<int>("covarianceVersion")),
      covariancePackingSchemas_(iConfig.getParameter<std::vector<int>>("covariancePackingSchemas")),
      muons_(consumes<reco::MuonCollection>(iConfig.getParameter<edm::InputTag>("muons"))),
      passThroughCut_(iConfig.getParameter<std::string>("passThroughCut")),
      maxDzForPrimaryAssignment_(
          iConfig.getParameter<edm::ParameterSet>("pvAssignment").getParameter<double>("maxDzForPrimaryAssignment")),
      maxDzSigForPrimaryAssignment_(
          iConfig.getParameter<edm::ParameterSet>("pvAssignment").getParameter<double>("maxDzSigForPrimaryAssignment")),
      maxDzErrorForPrimaryAssignment_(iConfig.getParameter<edm::ParameterSet>("pvAssignment")
                                          .getParameter<double>("maxDzErrorForPrimaryAssignment")),
      maxDxyForNotReconstructedPrimary_(iConfig.getParameter<edm::ParameterSet>("pvAssignment")
                                            .getParameter<double>("maxDxyForNotReconstructedPrimary")),
      maxDxySigForNotReconstructedPrimary_(iConfig.getParameter<edm::ParameterSet>("pvAssignment")
                                               .getParameter<double>("maxDxySigForNotReconstructedPrimary")),
      useLegacySetup_(iConfig.getParameter<bool>("useLegacySetup")) {
  std::vector<std::string> trkQuals(iConfig.getParameter<std::vector<std::string>>("qualsToAutoAccept"));
  std::transform(
      trkQuals.begin(), trkQuals.end(), std::back_inserter(qualsToAutoAccept_), reco::TrackBase::qualityByName);

  if (std::find(qualsToAutoAccept_.begin(), qualsToAutoAccept_.end(), reco::TrackBase::undefQuality) !=
      qualsToAutoAccept_.end()) {
    std::ostringstream msg;
    msg << " PATLostTracks has a quality requirement which resolves to undefQuality. This usually means a typo and is "
           "therefore treated a config error\nquality requirements:\n   ";
    for (const auto& trkQual : trkQuals)
      msg << trkQual << " ";
    throw cms::Exception("Configuration") << msg.str();
  }

  produces<std::vector<reco::Track>>();
  produces<std::vector<pat::PackedCandidate>>();
  produces<std::vector<pat::PackedCandidate>>("eleTracks");
  produces<edm::Association<pat::PackedCandidateCollection>>();
}

pat::PATLostTracks::~PATLostTracks() {}

void pat::PATLostTracks::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  edm::Handle<reco::PFCandidateCollection> cands;
  iEvent.getByToken(cands_, cands);

  edm::Handle<edm::Association<pat::PackedCandidateCollection>> pf2pc;
  iEvent.getByToken(map_, pf2pc);

  edm::Handle<reco::TrackCollection> tracks;
  iEvent.getByToken(tracks_, tracks);

  edm::Handle<reco::VertexCollection> vertices;
  iEvent.getByToken(vertices_, vertices);

  edm::Handle<reco::MuonCollection> muons;
  iEvent.getByToken(muons_, muons);

  edm::Handle<reco::VertexCompositeCandidateCollection> kshorts;
  iEvent.getByToken(kshorts_, kshorts);
  edm::Handle<reco::VertexCompositeCandidateCollection> lambdas;
  iEvent.getByToken(lambdas_, lambdas);

  edm::Handle<reco::VertexCollection> pvs;
  iEvent.getByToken(pv_, pvs);
  reco::VertexRef pv(pvs.id());
  reco::VertexRefProd pvRefProd(pvs);
  edm::Handle<reco::VertexCollection> pvOrigs;
  iEvent.getByToken(pvOrigs_, pvOrigs);

  auto outPtrTrks = std::make_unique<std::vector<reco::Track>>();
  auto outPtrTrksAsCands = std::make_unique<std::vector<pat::PackedCandidate>>();
  auto outPtrEleTrksAsCands = std::make_unique<std::vector<pat::PackedCandidate>>();

  std::vector<TrkStatus> trkStatus(tracks->size(), TrkStatus::NOTUSED);
  //Mark all tracks used in candidates
  //check if packed candidates are storing the tracks by seeing if number of hits >0
  //currently we dont use that information though
  //electrons will never store their track (they store the GSF track)
  for (unsigned int ic = 0, nc = cands->size(); ic < nc; ++ic) {
    edm::Ref<reco::PFCandidateCollection> r(cands, ic);
    const reco::PFCandidate& cand = (*cands)[ic];
    if (cand.charge() && cand.trackRef().isNonnull() && cand.trackRef().id() == tracks.id()) {
      if (cand.pdgId() == 11)
        trkStatus[cand.trackRef().key()] = TrkStatus::PFELECTRON;
      else if (cand.pdgId() == -11)
        trkStatus[cand.trackRef().key()] = TrkStatus::PFPOSITRON;
      else if ((*pf2pc)[r]->numberOfHits() > 0)
        trkStatus[cand.trackRef().key()] = TrkStatus::PFCAND;
      else
        trkStatus[cand.trackRef().key()] = TrkStatus::PFCANDNOTRKPROPS;
    }
  }

  //Mark all tracks used in secondary vertices
  for (const auto& secVert : *vertices) {
    for (auto trkIt = secVert.tracks_begin(); trkIt != secVert.tracks_end(); trkIt++) {
      if (trkStatus[trkIt->key()] == TrkStatus::NOTUSED)
        trkStatus[trkIt->key()] = TrkStatus::VTX;
    }
  }
  for (const auto& v0 : *kshorts) {
    for (size_t dIdx = 0; dIdx < v0.numberOfDaughters(); dIdx++) {
      size_t key = (dynamic_cast<const reco::RecoChargedCandidate*>(v0.daughter(dIdx)))->track().key();
      if (trkStatus[key] == TrkStatus::NOTUSED)
        trkStatus[key] = TrkStatus::VTX;
    }
  }
  for (const auto& v0 : *lambdas) {
    for (size_t dIdx = 0; dIdx < v0.numberOfDaughters(); dIdx++) {
      size_t key = (dynamic_cast<const reco::RecoChargedCandidate*>(v0.daughter(dIdx)))->track().key();
      if (trkStatus[key] == TrkStatus::NOTUSED)
        trkStatus[key] = TrkStatus::VTX;
    }
  }
  std::vector<int> mapping(tracks->size(), -1);
  int lostTrkIndx = 0;
  for (unsigned int trkIndx = 0; trkIndx < tracks->size(); trkIndx++) {
    reco::TrackRef trk(tracks, trkIndx);
    if (trkStatus[trkIndx] == TrkStatus::VTX || (trkStatus[trkIndx] == TrkStatus::NOTUSED && passTrkCuts(*trk))) {
      outPtrTrks->emplace_back(*trk);
      //association to PV
      std::pair<int, pat::PackedCandidate::PVAssociationQuality> pvAsso = associateTrkToVtx(*pvOrigs, trk);
      const reco::VertexRef& pvOrigRef = reco::VertexRef(pvOrigs, pvAsso.first);
      if (pvOrigRef.isNonnull()) {
        pv = reco::VertexRef(pvs, pvOrigRef.key());  // WARNING: assume the PV slimmer is keeping same order
      } else if (!pvs->empty()) {
        pv = reco::VertexRef(pvs, 0);
      }
      addPackedCandidate(*outPtrTrksAsCands, trk, pv, pvRefProd, trkStatus[trkIndx], pvAsso.second, muons);

      //for creating the reco::Track -> pat::PackedCandidate map
      //not done for the lostTrack:eleTracks collection
      mapping[trkIndx] = lostTrkIndx;
      lostTrkIndx++;
    } else if ((trkStatus[trkIndx] == TrkStatus::PFELECTRON || trkStatus[trkIndx] == TrkStatus::PFPOSITRON) &&
               passTrkCuts(*trk)) {
      //association to PV
      std::pair<int, pat::PackedCandidate::PVAssociationQuality> pvAsso = associateTrkToVtx(*pvOrigs, trk);
      const reco::VertexRef& pvOrigRef = reco::VertexRef(pvOrigs, pvAsso.first);
      if (pvOrigRef.isNonnull()) {
        pv = reco::VertexRef(pvs, pvOrigRef.key());  // WARNING: assume the PV slimmer is keeping same order
      } else if (!pvs->empty()) {
        pv = reco::VertexRef(pvs, 0);
      }
      addPackedCandidate(*outPtrEleTrksAsCands, trk, pv, pvRefProd, trkStatus[trkIndx], pvAsso.second, muons);
    }
  }

  iEvent.put(std::move(outPtrTrks));
  iEvent.put(std::move(outPtrEleTrksAsCands), "eleTracks");
  edm::OrphanHandle<pat::PackedCandidateCollection> oh = iEvent.put(std::move(outPtrTrksAsCands));
  auto tk2pc = std::make_unique<edm::Association<pat::PackedCandidateCollection>>(oh);
  edm::Association<pat::PackedCandidateCollection>::Filler tk2pcFiller(*tk2pc);
  tk2pcFiller.insert(tracks, mapping.begin(), mapping.end());
  tk2pcFiller.fill();
  iEvent.put(std::move(tk2pc));
}

bool pat::PATLostTracks::passTrkCuts(const reco::Track& tr) const {
  const bool passTrkHits = tr.pt() > minPt_ && tr.numberOfValidHits() >= minHits_ &&
                           tr.hitPattern().numberOfValidPixelHits() >= minPixelHits_;
  const bool passTrkQual = passesQuality(tr, qualsToAutoAccept_);

  return passTrkHits || passTrkQual || passThroughCut_(tr);
}

void pat::PATLostTracks::addPackedCandidate(std::vector<pat::PackedCandidate>& cands,
                                            const reco::TrackRef& trk,
                                            const reco::VertexRef& pvSlimmed,
                                            const reco::VertexRefProd& pvSlimmedColl,
                                            const pat::PATLostTracks::TrkStatus& trkStatus,
                                            const pat::PackedCandidate::PVAssociationQuality& pvAssocQuality,
                                            edm::Handle<reco::MuonCollection> muons) const {
  const float mass = 0.13957018;

  int id = 211 * trk->charge();
  if (trkStatus == TrkStatus::PFELECTRON)
    id = 11;
  else if (trkStatus == TrkStatus::PFPOSITRON)
    id = -11;

  // assign the proper pdgId for tracks that are reconstructed as a muon
  for (auto& mu : *muons) {
    if (reco::TrackRef(mu.innerTrack()) == trk) {
      id = -13 * trk->charge();
      break;
    }
  }

  reco::Candidate::PolarLorentzVector p4(trk->pt(), trk->eta(), trk->phi(), mass);
  cands.emplace_back(
      pat::PackedCandidate(p4, trk->vertex(), trk->pt(), trk->eta(), trk->phi(), id, pvSlimmedColl, pvSlimmed.key()));

  cands.back().setTrackHighPurity(trk->quality(reco::TrackBase::highPurity));

  if (trk->pt() > minPtToStoreProps_ || trkStatus == TrkStatus::VTX) {
    if (useLegacySetup_ || std::abs(id) == 11 || trkStatus == TrkStatus::VTX) {
      cands.back().setTrackProperties(*trk, covariancePackingSchemas_[4], covarianceVersion_);
    } else {
      if (trk->hitPattern().numberOfValidPixelHits() > 0) {
        cands.back().setTrackProperties(
            *trk, covariancePackingSchemas_[0], covarianceVersion_);  // high quality with pixels
      } else {
        cands.back().setTrackProperties(
            *trk, covariancePackingSchemas_[1], covarianceVersion_);  // high quality without pixels
      }
    }
  } else if (!useLegacySetup_ && trk->pt() > 0.5) {
    if (trk->hitPattern().numberOfValidPixelHits() > 0) {
      cands.back().setTrackProperties(
          *trk, covariancePackingSchemas_[2], covarianceVersion_);  // low quality with pixels
    } else {
      cands.back().setTrackProperties(
          *trk, covariancePackingSchemas_[3], covarianceVersion_);  // low quality without pixels
    }
  }
  cands.back().setAssociationQuality(pvAssocQuality);
}

std::pair<int, pat::PackedCandidate::PVAssociationQuality> pat::PATLostTracks::associateTrkToVtx(
    const reco::VertexCollection& vertices, const reco::TrackRef& trk) const {
  //For legacy setup check only if the track is used in fit of the PV, i.e. vertices[0],
  //and associate quality if weight > 0.5. Otherwise return invalid vertex index (-1)
  //and default quality flag (NotReconstructedPrimary = 0)
  if (useLegacySetup_) {
    float w = vertices[0].trackWeight(trk);
    if (w > 0.5) {
      return std::pair<int, pat::PackedCandidate::PVAssociationQuality>(0, pat::PackedCandidate::UsedInFitTight);
    } else if (w > 0.) {
      return std::pair<int, pat::PackedCandidate::PVAssociationQuality>(0,
                                                                        pat::PackedCandidate::NotReconstructedPrimary);
    } else {
      return std::pair<int, pat::PackedCandidate::PVAssociationQuality>(-1,
                                                                        pat::PackedCandidate::NotReconstructedPrimary);
    }
  }

  //Inspired by CommonTools/RecoAlgos/interface/PrimaryVertexAssignment.h
  //but without specific association for secondaries in jets and option to use timing

  int iVtxMaxWeight = -1;
  int iVtxMinDzDist = -1;
  size_t idx = 0;
  float maxWeight = 0;
  double minDz = std::numeric_limits<double>::max();
  double minDzSig = std::numeric_limits<double>::max();
  for (auto const& vtx : vertices) {
    float w = vtx.trackWeight(trk);
    double dz = std::abs(trk->dz(vtx.position()));
    double dzSig = dz / trk->dzError();
    if (w > maxWeight) {
      maxWeight = w;
      iVtxMaxWeight = idx;
    }
    if (dzSig < minDzSig) {
      minDzSig = dzSig;
      minDz = dz;
      iVtxMinDzDist = idx;
    }
    idx++;
  }
  // vertex in which fit the track was used
  if (iVtxMaxWeight >= 0) {
    if (maxWeight > 0.5) {
      return std::pair<int, pat::PackedCandidate::PVAssociationQuality>(iVtxMaxWeight,
                                                                        pat::PackedCandidate::UsedInFitTight);
    } else {
      return std::pair<int, pat::PackedCandidate::PVAssociationQuality>(iVtxMaxWeight,
                                                                        pat::PackedCandidate::UsedInFitLoose);
    }
  }
  // vertex "closest in Z" with tight cuts (targetting primary particles)
  if (minDz < maxDzForPrimaryAssignment_) {
    const double add_cov = vertices[iVtxMinDzDist].covariance(2, 2);
    const double dzErr = sqrt(trk->dzError() * trk->dzError() + add_cov);
    if (minDz / dzErr < maxDzSigForPrimaryAssignment_ && trk->dzError() < maxDzErrorForPrimaryAssignment_) {
      return std::pair<int, pat::PackedCandidate::PVAssociationQuality>(iVtxMinDzDist,
                                                                        pat::PackedCandidate::CompatibilityDz);
    }
  }
  // if the track is not compatible with other PVs but is compatible with the BeamSpot, we may simply have not reco'ed the PV!
  //  we still point it to the closest in Z, but flag it as possible orphan-primary
  if (!vertices.empty() && std::abs(trk->dxy(vertices[0].position())) < maxDxyForNotReconstructedPrimary_ &&
      std::abs(trk->dxy(vertices[0].position()) / trk->dxyError()) < maxDxySigForNotReconstructedPrimary_)
    return std::pair<int, pat::PackedCandidate::PVAssociationQuality>(iVtxMinDzDist,
                                                                      pat::PackedCandidate::NotReconstructedPrimary);
  // for tracks not associated to any PV return the closest in dz
  return std::pair<int, pat::PackedCandidate::PVAssociationQuality>(iVtxMinDzDist, pat::PackedCandidate::OtherDeltaZ);
}

using pat::PATLostTracks;
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PATLostTracks);
