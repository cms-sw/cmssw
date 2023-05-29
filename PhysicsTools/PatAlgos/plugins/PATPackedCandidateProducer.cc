#include <string>

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/PatCandidates/interface/HcalDepthEnergyFractions.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

/*#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoVertex/VertexPrimitives/interface/ConvertToFromReco.h"
#include
"TrackingTools/GeomPropagators/interface/AnalyticalImpactPointExtrapolator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
*/
//#define CRAZYSORT

namespace pat {
  /// conversion map from quality flags used in PV association and miniAOD one
  const static int qualityMap[8] = {1, 0, 1, 1, 4, 4, 5, 6};

  class PATPackedCandidateProducer : public edm::global::EDProducer<> {
  public:
    explicit PATPackedCandidateProducer(const edm::ParameterSet &);
    ~PATPackedCandidateProducer() override;

    void produce(edm::StreamID, edm::Event &, const edm::EventSetup &) const override;

    // sorting of cands to maximize the zlib compression
    static bool candsOrdering(pat::PackedCandidate const &i, pat::PackedCandidate const &j) {
      if (std::abs(i.charge()) == std::abs(j.charge())) {
        if (i.charge() != 0) {
          if (i.hasTrackDetails() and !j.hasTrackDetails())
            return true;
          if (!i.hasTrackDetails() and j.hasTrackDetails())
            return false;
          if (i.covarianceSchema() > j.covarianceSchema())
            return true;
          if (i.covarianceSchema() < j.covarianceSchema())
            return false;
        }
        if (i.vertexRef() == j.vertexRef())
          return i.eta() > j.eta();
        else
          return i.vertexRef().key() < j.vertexRef().key();
      }
      return std::abs(i.charge()) > std::abs(j.charge());
    }

    template <typename T>
    static std::vector<size_t> sort_indexes(const std::vector<T> &v) {
      std::vector<size_t> idx(v.size());
      for (size_t i = 0; i != idx.size(); ++i)
        idx[i] = i;
      std::sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) { return candsOrdering(v[i1], v[i2]); });
      return idx;
    }

  private:
    // if PuppiSrc && PuppiNoLepSrc are empty, usePuppi is false
    // otherwise assumes that if they are set, you wanted to use puppi and will
    // throw an exception if the puppis are not found
    const bool usePuppi_;

    const edm::EDGetTokenT<reco::PFCandidateCollection> Cands_;
    const edm::EDGetTokenT<reco::VertexCollection> PVs_;
    const edm::EDGetTokenT<edm::Association<reco::VertexCollection>> PVAsso_;
    const edm::EDGetTokenT<edm::ValueMap<int>> PVAssoQuality_;
    const edm::EDGetTokenT<reco::VertexCollection> PVOrigs_;
    const edm::EDGetTokenT<reco::TrackCollection> TKOrigs_;
    const edm::EDGetTokenT<edm::ValueMap<float>> PuppiWeight_;
    const edm::EDGetTokenT<edm::ValueMap<float>> PuppiWeightNoLep_;
    std::vector<edm::EDGetTokenT<edm::View<reco::Candidate>>> SVWhiteLists_;
    const bool storeChargedHadronIsolation_;
    const edm::EDGetTokenT<edm::ValueMap<bool>> ChargedHadronIsolation_;

    const double minPtForChargedHadronProperties_;
    const double minPtForTrackProperties_;
    const double minPtForLowQualityTrackProperties_;
    const int covarianceVersion_;
    const std::vector<int> covariancePackingSchemas_;

    const std::vector<int> pfCandidateTypesForHcalDepth_;
    const bool storeHcalDepthEndcapOnly_;

    const bool storeTiming_;
    const bool timeFromValueMap_;
    const edm::EDGetTokenT<edm::ValueMap<float>> t0Map_;
    const edm::EDGetTokenT<edm::ValueMap<float>> t0ErrMap_;

    // for debugging
    float calcDxy(float dx, float dy, float phi) const { return -dx * std::sin(phi) + dy * std::cos(phi); }
    float calcDz(reco::Candidate::Point p, reco::Candidate::Point v, const reco::Candidate &c) const {
      return p.Z() - v.Z() - ((p.X() - v.X()) * c.px() + (p.Y() - v.Y()) * c.py()) * c.pz() / (c.pt() * c.pt());
    }
  };
}  // namespace pat

pat::PATPackedCandidateProducer::PATPackedCandidateProducer(const edm::ParameterSet &iConfig)
    : usePuppi_(!iConfig.getParameter<edm::InputTag>("PuppiSrc").encode().empty() ||
                !iConfig.getParameter<edm::InputTag>("PuppiNoLepSrc").encode().empty()),
      Cands_(consumes<reco::PFCandidateCollection>(iConfig.getParameter<edm::InputTag>("inputCollection"))),
      PVs_(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("inputVertices"))),
      PVAsso_(
          consumes<edm::Association<reco::VertexCollection>>(iConfig.getParameter<edm::InputTag>("vertexAssociator"))),
      PVAssoQuality_(consumes<edm::ValueMap<int>>(iConfig.getParameter<edm::InputTag>("vertexAssociator"))),
      PVOrigs_(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("originalVertices"))),
      TKOrigs_(consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("originalTracks"))),
      PuppiWeight_(usePuppi_ ? consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("PuppiSrc"))
                             : edm::EDGetTokenT<edm::ValueMap<float>>()),
      PuppiWeightNoLep_(usePuppi_ ? consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("PuppiNoLepSrc"))
                                  : edm::EDGetTokenT<edm::ValueMap<float>>()),
      storeChargedHadronIsolation_(!iConfig.getParameter<edm::InputTag>("chargedHadronIsolation").encode().empty()),
      ChargedHadronIsolation_(
          consumes<edm::ValueMap<bool>>(iConfig.getParameter<edm::InputTag>("chargedHadronIsolation"))),
      minPtForChargedHadronProperties_(iConfig.getParameter<double>("minPtForChargedHadronProperties")),
      minPtForTrackProperties_(iConfig.getParameter<double>("minPtForTrackProperties")),
      minPtForLowQualityTrackProperties_(iConfig.getParameter<double>("minPtForLowQualityTrackProperties")),
      covarianceVersion_(iConfig.getParameter<int>("covarianceVersion")),
      covariancePackingSchemas_(iConfig.getParameter<std::vector<int>>("covariancePackingSchemas")),
      pfCandidateTypesForHcalDepth_(iConfig.getParameter<std::vector<int>>("pfCandidateTypesForHcalDepth")),
      storeHcalDepthEndcapOnly_(iConfig.getParameter<bool>("storeHcalDepthEndcapOnly")),
      storeTiming_(iConfig.getParameter<bool>("storeTiming")),
      timeFromValueMap_(!iConfig.getParameter<edm::InputTag>("timeMap").encode().empty() &&
                        !iConfig.getParameter<edm::InputTag>("timeMapErr").encode().empty()),
      t0Map_(timeFromValueMap_ ? consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("timeMap"))
                               : edm::EDGetTokenT<edm::ValueMap<float>>()),
      t0ErrMap_(timeFromValueMap_ ? consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("timeMapErr"))
                                  : edm::EDGetTokenT<edm::ValueMap<float>>()) {
  std::vector<edm::InputTag> sv_tags =
      iConfig.getParameter<std::vector<edm::InputTag>>("secondaryVerticesForWhiteList");
  for (const auto &itag : sv_tags) {
    SVWhiteLists_.push_back(consumes<edm::View<reco::Candidate>>(itag));
  }

  produces<std::vector<pat::PackedCandidate>>();
  produces<edm::Association<pat::PackedCandidateCollection>>();
  produces<edm::Association<reco::PFCandidateCollection>>();

  if (not pfCandidateTypesForHcalDepth_.empty())
    produces<edm::ValueMap<pat::HcalDepthEnergyFractions>>("hcalDepthEnergyFractions");
}

pat::PATPackedCandidateProducer::~PATPackedCandidateProducer() {}

void pat::PATPackedCandidateProducer::produce(edm::StreamID, edm::Event &iEvent, const edm::EventSetup &iSetup) const {
  edm::Handle<reco::PFCandidateCollection> cands;
  iEvent.getByToken(Cands_, cands);

  edm::Handle<edm::ValueMap<float>> puppiWeight;
  edm::Handle<edm::ValueMap<float>> puppiWeightNoLep;
  if (usePuppi_) {
    iEvent.getByToken(PuppiWeight_, puppiWeight);
    iEvent.getByToken(PuppiWeightNoLep_, puppiWeightNoLep);
  }

  edm::Handle<reco::VertexCollection> PVOrigs;
  iEvent.getByToken(PVOrigs_, PVOrigs);

  edm::Handle<edm::Association<reco::VertexCollection>> assoHandle;
  iEvent.getByToken(PVAsso_, assoHandle);
  edm::Handle<edm::ValueMap<int>> assoQualityHandle;
  iEvent.getByToken(PVAssoQuality_, assoQualityHandle);
  const edm::Association<reco::VertexCollection> &associatedPV = *(assoHandle.product());
  const edm::ValueMap<int> &associationQuality = *(assoQualityHandle.product());

  edm::Handle<edm::ValueMap<bool>> chargedHadronIsolationHandle;
  if (storeChargedHadronIsolation_)
    iEvent.getByToken(ChargedHadronIsolation_, chargedHadronIsolationHandle);

  std::set<unsigned int> whiteList;
  std::set<reco::TrackRef> whiteListTk;
  for (auto itoken : SVWhiteLists_) {
    edm::Handle<edm::View<reco::Candidate>> svWhiteListHandle;
    iEvent.getByToken(itoken, svWhiteListHandle);
    const edm::View<reco::Candidate> &svWhiteList = *(svWhiteListHandle.product());
    for (unsigned int i = 0; i < svWhiteList.size(); i++) {
      // Whitelist via Ptrs
      for (unsigned int j = 0; j < svWhiteList[i].numberOfSourceCandidatePtrs(); j++) {
        const edm::Ptr<reco::Candidate> &c = svWhiteList[i].sourceCandidatePtr(j);
        if (c.id() == cands.id())
          whiteList.insert(c.key());
      }
      // Whitelist via RecoCharged
      for (auto dau = svWhiteList[i].begin(); dau != svWhiteList[i].end(); dau++) {
        const reco::RecoChargedCandidate *chCand = dynamic_cast<const reco::RecoChargedCandidate *>(&(*dau));
        if (chCand != nullptr) {
          whiteListTk.insert(chCand->track());
        }
      }
    }
  }

  edm::Handle<edm::ValueMap<float>> t0Map;
  edm::Handle<edm::ValueMap<float>> t0ErrMap;
  if (timeFromValueMap_) {
    iEvent.getByToken(t0Map_, t0Map);
    iEvent.getByToken(t0ErrMap_, t0ErrMap);
  }

  edm::Handle<reco::VertexCollection> PVs;
  iEvent.getByToken(PVs_, PVs);
  reco::VertexRef PV(PVs.id());
  reco::VertexRefProd PVRefProd(PVs);
  math::XYZPoint PVpos;

  std::vector<pat::HcalDepthEnergyFractions> hcalDepthEnergyFractions;
  hcalDepthEnergyFractions.reserve(cands->size());
  std::vector<pat::HcalDepthEnergyFractions> hcalDepthEnergyFractions_Ordered;
  hcalDepthEnergyFractions_Ordered.reserve(cands->size());

  edm::Handle<reco::TrackCollection> TKOrigs;
  iEvent.getByToken(TKOrigs_, TKOrigs);
  auto outPtrP = std::make_unique<std::vector<pat::PackedCandidate>>();
  std::vector<int> mapping(cands->size());
  std::vector<int> mappingReverse(cands->size());
  std::vector<int> mappingTk(TKOrigs->size(), -1);

  for (unsigned int ic = 0, nc = cands->size(); ic < nc; ++ic) {
    const reco::PFCandidate &cand = (*cands)[ic];
    const reco::Track *ctrack = nullptr;
    if ((abs(cand.pdgId()) == 11 || cand.pdgId() == 22) && cand.gsfTrackRef().isNonnull()) {
      ctrack = &*cand.gsfTrackRef();
    } else if (cand.trackRef().isNonnull()) {
      ctrack = &*cand.trackRef();
    }
    if (ctrack) {
      float dist = 1e99;
      int pvi = -1;
      for (size_t ii = 0; ii < PVs->size(); ii++) {
        float dz = std::abs(ctrack->dz(((*PVs)[ii]).position()));
        if (dz < dist) {
          pvi = ii;
          dist = dz;
        }
      }
      PV = reco::VertexRef(PVs, pvi);
      math::XYZPoint vtx = cand.vertex();
      pat::PackedCandidate::LostInnerHits lostHits = pat::PackedCandidate::noLostInnerHits;
      const reco::VertexRef &PVOrig = associatedPV[reco::CandidatePtr(cands, ic)];
      if (PVOrig.isNonnull())
        PV = reco::VertexRef(PVs,
                             PVOrig.key());  // WARNING: assume the PV slimmer is keeping same order
      int quality = associationQuality[reco::CandidatePtr(cands, ic)];
      //          if ((size_t)pvi!=PVOrig.key()) std::cout << "not closest in Z"
      //          << pvi << " " << PVOrig.key() << " " << cand.pt() << " " <<
      //          quality << std::endl; TrajectoryStateOnSurface tsos =
      //          extrapolator.extrapolate(trajectoryStateTransform::initialFreeState(*ctrack,&*magneticField),
      //          RecoVertex::convertPos(PV->position()));
      //   vtx = tsos.globalPosition();
      //          phiAtVtx = tsos.globalDirection().phi();
      vtx = ctrack->referencePoint();
      float ptTrk = ctrack->pt();
      float etaAtVtx = ctrack->eta();
      float phiAtVtx = ctrack->phi();

      int nlost = ctrack->hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS);
      if (nlost == 0) {
        if (ctrack->hitPattern().hasValidHitInPixelLayer(PixelSubdetector::SubDetector::PixelBarrel, 1)) {
          lostHits = pat::PackedCandidate::validHitInFirstPixelBarrelLayer;
        }
      } else {
        lostHits = (nlost == 1 ? pat::PackedCandidate::oneLostInnerHit : pat::PackedCandidate::moreLostInnerHits);
      }

      outPtrP->push_back(
          pat::PackedCandidate(cand.polarP4(), vtx, ptTrk, etaAtVtx, phiAtVtx, cand.pdgId(), PVRefProd, PV.key()));
      outPtrP->back().setAssociationQuality(pat::PackedCandidate::PVAssociationQuality(qualityMap[quality]));
      outPtrP->back().setCovarianceVersion(covarianceVersion_);
      if (cand.trackRef().isNonnull() && PVOrig.isNonnull() && PVOrig->trackWeight(cand.trackRef()) > 0.5 &&
          quality == 7) {
        outPtrP->back().setAssociationQuality(pat::PackedCandidate::UsedInFitTight);
      }
      // properties of the best track
      outPtrP->back().setLostInnerHits(lostHits);
      if (outPtrP->back().pt() > minPtForTrackProperties_ || outPtrP->back().ptTrk() > minPtForTrackProperties_ ||
          whiteList.find(ic) != whiteList.end() ||
          (cand.trackRef().isNonnull() && whiteListTk.find(cand.trackRef()) != whiteListTk.end())) {
        outPtrP->back().setTrkAlgo(static_cast<uint8_t>(ctrack->algo()), static_cast<uint8_t>(ctrack->originalAlgo()));
        outPtrP->back().setFirstHit(ctrack->hitPattern().getHitPattern(reco::HitPattern::TRACK_HITS, 0));
        if (abs(outPtrP->back().pdgId()) == 22) {
          outPtrP->back().setTrackProperties(*ctrack, covariancePackingSchemas_[4], covarianceVersion_);
        } else {
          if (ctrack->hitPattern().numberOfValidPixelHits() > 0) {
            outPtrP->back().setTrackProperties(*ctrack,
                                               covariancePackingSchemas_[0],
                                               covarianceVersion_);  // high quality
          } else {
            outPtrP->back().setTrackProperties(*ctrack, covariancePackingSchemas_[1], covarianceVersion_);
          }
        }
        // outPtrP->back().setTrackProperties(*ctrack,tsos.curvilinearError());
      } else {
        if (outPtrP->back().pt() > minPtForLowQualityTrackProperties_) {
          if (ctrack->hitPattern().numberOfValidPixelHits() > 0)
            outPtrP->back().setTrackProperties(*ctrack,
                                               covariancePackingSchemas_[2],
                                               covarianceVersion_);  // low quality, with pixels
          else
            outPtrP->back().setTrackProperties(*ctrack,
                                               covariancePackingSchemas_[3],
                                               covarianceVersion_);  // low quality, without pixels
        }
      }

      // these things are always for the CKF track
      outPtrP->back().setTrackHighPurity(cand.trackRef().isNonnull() &&
                                         cand.trackRef()->quality(reco::Track::highPurity));
      if (cand.muonRef().isNonnull()) {
        outPtrP->back().setMuonID(cand.muonRef()->isStandAloneMuon(), cand.muonRef()->isGlobalMuon());
      }
    } else {
      if (!PVs->empty()) {
        PV = reco::VertexRef(PVs, 0);
        PVpos = PV->position();
      }

      outPtrP->push_back(pat::PackedCandidate(
          cand.polarP4(), PVpos, cand.pt(), cand.eta(), cand.phi(), cand.pdgId(), PVRefProd, PV.key()));
      outPtrP->back().setAssociationQuality(
          pat::PackedCandidate::PVAssociationQuality(pat::PackedCandidate::UsedInFitTight));
    }

    // neutrals and isolated charged hadrons

    bool isIsolatedChargedHadron = false;
    if (storeChargedHadronIsolation_) {
      const edm::ValueMap<bool> &chargedHadronIsolation = *(chargedHadronIsolationHandle.product());
      isIsolatedChargedHadron =
          ((cand.pt() > minPtForChargedHadronProperties_) && (chargedHadronIsolation[reco::PFCandidateRef(cands, ic)]));
      outPtrP->back().setIsIsolatedChargedHadron(isIsolatedChargedHadron);
    }

    if (abs(cand.pdgId()) == 1 || abs(cand.pdgId()) == 130) {
      outPtrP->back().setHcalFraction(cand.hcalEnergy() / (cand.ecalEnergy() + cand.hcalEnergy()));
    } else if ((cand.charge() || abs(cand.pdgId()) == 22) && cand.pt() > 0.5) {
      outPtrP->back().setHcalFraction(cand.hcalEnergy() / (cand.ecalEnergy() + cand.hcalEnergy()));
      outPtrP->back().setCaloFraction((cand.hcalEnergy() + cand.ecalEnergy()) / cand.energy());
    } else {
      outPtrP->back().setHcalFraction(0);
      outPtrP->back().setCaloFraction(0);
    }

    if (isIsolatedChargedHadron) {
      outPtrP->back().setRawCaloFraction((cand.rawEcalEnergy() + cand.rawHcalEnergy()) / cand.energy());
      outPtrP->back().setRawHcalFraction(cand.rawHcalEnergy() / (cand.rawEcalEnergy() + cand.rawHcalEnergy()));
    } else {
      outPtrP->back().setRawCaloFraction(0);
      outPtrP->back().setRawHcalFraction(0);
    }

    std::vector<float> dummyVector;
    dummyVector.clear();
    pat::HcalDepthEnergyFractions hcalDepthEFrac(dummyVector);

    // storing HcalDepthEnergyFraction information
    if (std::find(pfCandidateTypesForHcalDepth_.begin(), pfCandidateTypesForHcalDepth_.end(), abs(cand.pdgId())) !=
        pfCandidateTypesForHcalDepth_.end()) {
      if (!storeHcalDepthEndcapOnly_ ||
          fabs(outPtrP->back().eta()) > 1.3) {  // storeHcalDepthEndcapOnly_==false -> store all eta of
                                                // selected PF types, if true, only |eta|>1.3 of selected
                                                // PF types will be stored
        std::vector<float> hcalDepthEnergyFractionTmp(cand.hcalDepthEnergyFractions().begin(),
                                                      cand.hcalDepthEnergyFractions().end());
        hcalDepthEFrac.reset(hcalDepthEnergyFractionTmp);
      }
    }
    hcalDepthEnergyFractions.push_back(hcalDepthEFrac);

    // specifically this is the PFLinker requirements to apply the e/gamma
    // regression
    if (cand.particleId() == reco::PFCandidate::e ||
        (cand.particleId() == reco::PFCandidate::gamma && cand.mva_nothing_gamma() > 0.)) {
      outPtrP->back().setGoodEgamma();
    }

    if (usePuppi_) {
      reco::PFCandidateRef pkref(cands, ic);

      float puppiWeightVal = (*puppiWeight)[pkref];
      float puppiWeightNoLepVal = (*puppiWeightNoLep)[pkref];
      outPtrP->back().setPuppiWeight(puppiWeightVal, puppiWeightNoLepVal);
    }

    if (storeTiming_) {
      if (timeFromValueMap_) {
        if (cand.trackRef().isNonnull()) {
          auto t0 = (*t0Map)[cand.trackRef()];
          auto t0Err = (*t0ErrMap)[cand.trackRef()];
          outPtrP->back().setTime(t0, t0Err);
        }
      } else {
        if (cand.isTimeValid()) {
          outPtrP->back().setTime(cand.time(), cand.timeError());
        }
      }
    }

    mapping[ic] = ic;  // trivial at the moment!
    if (cand.trackRef().isNonnull() && cand.trackRef().id() == TKOrigs.id()) {
      mappingTk[cand.trackRef().key()] = ic;
    }
  }

  auto outPtrPSorted = std::make_unique<std::vector<pat::PackedCandidate>>();
  std::vector<size_t> order = sort_indexes(*outPtrP);
  std::vector<size_t> reverseOrder(order.size());
  for (size_t i = 0, nc = cands->size(); i < nc; i++) {
    outPtrPSorted->push_back((*outPtrP)[order[i]]);
    reverseOrder[order[i]] = i;
    mappingReverse[order[i]] = i;
    hcalDepthEnergyFractions_Ordered.push_back(hcalDepthEnergyFractions[order[i]]);
  }

  // Fix track association for sorted candidates
  for (size_t i = 0, ntk = mappingTk.size(); i < ntk; i++) {
    if (mappingTk[i] >= 0)
      mappingTk[i] = reverseOrder[mappingTk[i]];
  }

  edm::OrphanHandle<pat::PackedCandidateCollection> oh = iEvent.put(std::move(outPtrPSorted));

  // now build the two maps
  auto pf2pc = std::make_unique<edm::Association<pat::PackedCandidateCollection>>(oh);
  auto pc2pf = std::make_unique<edm::Association<reco::PFCandidateCollection>>(cands);
  edm::Association<pat::PackedCandidateCollection>::Filler pf2pcFiller(*pf2pc);
  edm::Association<reco::PFCandidateCollection>::Filler pc2pfFiller(*pc2pf);
  pf2pcFiller.insert(cands, mappingReverse.begin(), mappingReverse.end());
  pc2pfFiller.insert(oh, order.begin(), order.end());
  // include also the mapping track -> packed PFCand
  pf2pcFiller.insert(TKOrigs, mappingTk.begin(), mappingTk.end());

  pf2pcFiller.fill();
  pc2pfFiller.fill();
  iEvent.put(std::move(pf2pc));
  iEvent.put(std::move(pc2pf));

  // HCAL depth energy fraction additions using ValueMap
  auto hcalDepthEnergyFractionsV = std::make_unique<edm::ValueMap<HcalDepthEnergyFractions>>();
  edm::ValueMap<HcalDepthEnergyFractions>::Filler fillerHcalDepthEnergyFractions(*hcalDepthEnergyFractionsV);
  fillerHcalDepthEnergyFractions.insert(
      cands, hcalDepthEnergyFractions_Ordered.begin(), hcalDepthEnergyFractions_Ordered.end());
  fillerHcalDepthEnergyFractions.fill();

  if (not pfCandidateTypesForHcalDepth_.empty())
    iEvent.put(std::move(hcalDepthEnergyFractionsV), "hcalDepthEnergyFractions");
}

using pat::PATPackedCandidateProducer;
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PATPackedCandidateProducer);
