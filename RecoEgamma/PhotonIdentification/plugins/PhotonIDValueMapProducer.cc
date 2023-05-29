#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/PatCandidates/interface/Photon.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/isFinite.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "FWCore/Framework/interface/Event.h"

#include <memory>
#include <string>
#include <vector>

namespace {

  // This template function finds whether theCandidate is in thefootprint
  // collection. It is templated to be able to handle both reco and pat
  // photons (from AOD and miniAOD, respectively).
  template <class T>
  bool isInFootprint(const T& footprint, const edm::Ptr<reco::Candidate>& candidate) {
    for (auto& it : footprint) {
      if (it.key() == candidate.key())
        return true;
    }
    return false;
  }

  struct CachingPtrCandidate {
    CachingPtrCandidate(const reco::Candidate* cPtr, bool isAOD)
        : candidate(cPtr),
          track(isAOD ? &*static_cast<const reco::PFCandidate*>(cPtr)->trackRef() : nullptr),
          packed(isAOD ? nullptr : static_cast<const pat::PackedCandidate*>(cPtr)) {}

    const reco::Candidate* candidate;
    const reco::Track* track;
    const pat::PackedCandidate* packed;
  };

  void getImpactParameters(const CachingPtrCandidate& candidate, const reco::Vertex& pv, float& dxy, float& dz) {
    if (candidate.track != nullptr) {
      dxy = candidate.track->dxy(pv.position());
      dz = candidate.track->dz(pv.position());
    } else {
      dxy = candidate.packed->dxy(pv.position());
      dz = candidate.packed->dz(pv.position());
    }
  }

  // Some helper functions that are needed to access info in
  // AOD vs miniAOD
  reco::PFCandidate::ParticleType getCandidatePdgId(const reco::Candidate* candidate, bool isAOD) {
    if (isAOD)
      return static_cast<const reco::PFCandidate*>(candidate)->particleId();

    // the neutral hadrons and charged hadrons can be of pdgId types
    // only 130 (K0L) and +-211 (pi+-) in packed candidates
    const int pdgId = static_cast<const pat::PackedCandidate*>(candidate)->pdgId();
    if (pdgId == 22)
      return reco::PFCandidate::gamma;
    else if (abs(pdgId) == 130)  // PDG ID for K0L
      return reco::PFCandidate::h0;
    else if (abs(pdgId) == 211)  // PDG ID for pi+-
      return reco::PFCandidate::h;
    else
      return reco::PFCandidate::X;
  }

};  // namespace

class PhotonIDValueMapProducer : public edm::global::EDProducer<> {
public:
  explicit PhotonIDValueMapProducer(const edm::ParameterSet&);
  ~PhotonIDValueMapProducer() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  // This function computes charged hadron isolation with respect to multiple
  // PVs and returns the worst of the found isolation values. The function
  // implements the computation method taken directly from Run 1 code of
  // H->gamma gamma, specifically from the class CiCPhotonID of the
  // HiggsTo2photons anaysis code. Template is introduced to handle reco/pat
  // photons and aod/miniAOD PF candidates collections
  float computeWorstPFChargedIsolation(const reco::Photon& photon,
                                       const std::vector<edm::Ptr<reco::Candidate>>& pfCands,
                                       const reco::VertexCollection& vertices,
                                       const reco::Vertex& pv,
                                       unsigned char options,
                                       bool isAOD) const;

  // check whether a non-null preshower is there
  const bool usesES_;

  // Tokens
  const edm::EDGetTokenT<edm::View<reco::Photon>> src_;
  const edm::EDGetTokenT<EcalRecHitCollection> ebRecHits_;
  const edm::EDGetTokenT<EcalRecHitCollection> eeRecHits_;
  const edm::EDGetTokenT<EcalRecHitCollection> esRecHits_;
  const edm::EDGetTokenT<reco::VertexCollection> vtxToken_;
  const edm::EDGetTokenT<edm::View<reco::Candidate>> pfCandsToken_;
  const edm::EDGetToken particleBasedIsolationToken_;

  const EcalClusterLazyTools::ESGetTokens ecalClusterToolsESGetTokens_;

  const bool isAOD_;
};

constexpr int nVars_ = 19;

const std::string names[nVars_] = {
    // Cluster shapes
    "phoFull5x5SigmaIEtaIEta",  // 0
    "phoFull5x5SigmaIEtaIPhi",
    "phoFull5x5E1x3",
    "phoFull5x5E2x2",
    "phoFull5x5E2x5Max",
    "phoFull5x5E5x5",  // 5
    "phoESEffSigmaRR",
    // Cluster shape ratios
    "phoFull5x5E1x3byE5x5",
    "phoFull5x5E2x2byE5x5",
    "phoFull5x5E2x5byE5x5",
    // Isolations
    "phoChargedIsolation",  // 10
    "phoNeutralHadronIsolation",
    "phoPhotonIsolation",
    "phoWorstChargedIsolation",
    "phoWorstChargedIsolationConeVeto",
    "phoWorstChargedIsolationConeVetoPVConstr",  // 15
    // PFCluster Isolation
    "phoTrkIsolation",
    "phoHcalPFClIsolation",
    "phoEcalPFClIsolation"};

// options and bitflags
constexpr float coneSizeDR2 = 0.3 * 0.3;
constexpr float dxyMax = 0.1;
constexpr float dzMax = 0.2;
constexpr float dRveto2Barrel = 0.02 * 0.02;
constexpr float dRveto2Endcap = 0.02 * 0.02;
constexpr float ptMin = 0.1;

const unsigned char PV_CONSTRAINT = 0x1;
const unsigned char DR_VETO = 0x2;
const unsigned char PT_MIN_THRESH = 0x8;

PhotonIDValueMapProducer::PhotonIDValueMapProducer(const edm::ParameterSet& cfg)
    : usesES_(!cfg.getParameter<edm::InputTag>("esReducedRecHitCollection").label().empty()),
      src_(consumes(cfg.getParameter<edm::InputTag>("src"))),
      ebRecHits_(consumes(cfg.getParameter<edm::InputTag>("ebReducedRecHitCollection"))),
      eeRecHits_(consumes(cfg.getParameter<edm::InputTag>("eeReducedRecHitCollection"))),
      esRecHits_(consumes(cfg.getParameter<edm::InputTag>("esReducedRecHitCollection"))),
      vtxToken_(consumes(cfg.getParameter<edm::InputTag>("vertices"))),
      pfCandsToken_(consumes(cfg.getParameter<edm::InputTag>("pfCandidates"))),
      particleBasedIsolationToken_(mayConsume<edm::ValueMap<std::vector<reco::PFCandidateRef>>>(
          cfg.getParameter<edm::InputTag>("particleBasedIsolation")) /* ...only for AOD... */),
      ecalClusterToolsESGetTokens_{consumesCollector()},
      isAOD_(cfg.getParameter<bool>("isAOD")) {
  // Declare producibles
  for (int i = 0; i < nVars_; ++i)
    produces<edm::ValueMap<float>>(names[i]);
}

void PhotonIDValueMapProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  // Get the handles
  auto src = iEvent.getHandle(src_);
  auto vertices = iEvent.getHandle(vtxToken_);
  auto pfCandsHandle = iEvent.getHandle(pfCandsToken_);

  edm::Handle<edm::ValueMap<std::vector<reco::PFCandidateRef>>> particleBasedIsolationMap;
  if (isAOD_) {  // this exists only in AOD
    iEvent.getByToken(particleBasedIsolationToken_, particleBasedIsolationMap);
  } else if (!src->empty()) {
    edm::Ptr<pat::Photon> test(src->ptrAt(0));
    if (test.isNull() || !test.isAvailable()) {
      throw cms::Exception("InvalidConfiguration")
          << "DataFormat is detected as miniAOD but cannot cast to pat::Photon!";
    }
  }

  // Configure Lazy Tools, which will compute 5x5 quantities
  auto const& ecalClusterToolsESData = ecalClusterToolsESGetTokens_.get(iSetup);
  auto lazyToolnoZS =
      usesES_ ? noZS::EcalClusterLazyTools(iEvent, ecalClusterToolsESData, ebRecHits_, eeRecHits_, esRecHits_)
              : noZS::EcalClusterLazyTools(iEvent, ecalClusterToolsESData, ebRecHits_, eeRecHits_);

  // Get PV
  if (vertices->empty())
    return;  // skip the event if no PV found
  const reco::Vertex& pv = vertices->front();

  std::vector<float> vars[nVars_];

  std::vector<edm::Ptr<reco::Candidate>> pfCandNoNaN;
  for (const auto& pf : pfCandsHandle->ptrs()) {
    if (edm::isNotFinite(pf->pt())) {
      edm::LogWarning("PhotonIDValueMapProducer") << "PF candidate pT is NaN, skipping, see issue #39110" << std::endl;
    } else {
      pfCandNoNaN.push_back(pf);
    }
  }

  // reco::Photon::superCluster() is virtual so we can exploit polymorphism
  for (auto const& iPho : src->ptrs()) {
    //
    // Compute full 5x5 quantities
    //
    const auto& seed = *(iPho->superCluster()->seed());

    // For full5x5_sigmaIetaIeta, for 720 we use: lazy tools for AOD,
    // and userFloats or lazy tools for miniAOD. From some point in 72X and on, one can
    // retrieve the full5x5 directly from the object with ->full5x5_sigmaIetaIeta()
    // for both formats.
    const auto& vCov = lazyToolnoZS.localCovariances(seed);
    vars[0].push_back(edm::isNotFinite(vCov[0]) ? 0. : sqrt(vCov[0]));
    vars[1].push_back(vCov[1]);
    vars[2].push_back(lazyToolnoZS.e1x3(seed));
    vars[3].push_back(lazyToolnoZS.e2x2(seed));
    vars[4].push_back(lazyToolnoZS.e2x5Max(seed));
    vars[5].push_back(lazyToolnoZS.e5x5(seed));
    vars[6].push_back(lazyToolnoZS.eseffsirir(*(iPho->superCluster())));
    vars[7].push_back(vars[2].back() / vars[5].back());
    vars[8].push_back(vars[3].back() / vars[5].back());
    vars[9].push_back(vars[4].back() / vars[5].back());

    //
    // Compute absolute uncorrected isolations with footprint removal
    //

    // First, find photon direction with respect to the good PV
    math::XYZVector phoWrtVtx(
        iPho->superCluster()->x() - pv.x(), iPho->superCluster()->y() - pv.y(), iPho->superCluster()->z() - pv.z());

    // isolation sums
    float chargedIsoSum = 0.;
    float neutralHadronIsoSum = 0.;
    float photonIsoSum = 0.;

    // Loop over nan-free PF candidates
    for (auto const& iCand : pfCandNoNaN) {
      // Here, the type will be a simple reco::Candidate. We cast it
      // for full PFCandidate or PackedCandidate below as necessary

      // One would think that we should check that this iCand from the
      // generic PF collection is not identical to the iPho photon for
      // which we are computing the isolations. However, it turns out to
      // be unnecessary. Below, in the function isInFootprint(), we drop
      // this iCand if it is in the footprint, and this always removes
      // the iCand if it matches the iPho. The explicit check at this
      // point is not totally trivial because of non-triviality of
      // implementation of this check for miniAOD (PackedCandidates of
      // the PF collection do not contain the supercluser link, so can't
      // use that).
      //
      // if( isAOD_ ) {
      //     if( ((const edm::Ptr<reco::PFCandidate>)iCand)->superClusterRef() == iPho->superCluster() )
      //     continue;
      // }

      // Check if this candidate is within the isolation cone
      float dR2 = deltaR2(phoWrtVtx.Eta(), phoWrtVtx.Phi(), iCand->eta(), iCand->phi());
      if (dR2 > coneSizeDR2)
        continue;

      // Check if this candidate is not in the footprint
      if (isAOD_) {
        if (isInFootprint((*particleBasedIsolationMap)[iPho], iCand))
          continue;
      } else {
        edm::Ptr<pat::Photon> patPhotonPtr(iPho);
        if (isInFootprint(patPhotonPtr->associatedPackedPFCandidates(), iCand))
          continue;
      }

      // Find candidate type
      reco::PFCandidate::ParticleType thisCandidateType = getCandidatePdgId(&*iCand, isAOD_);

      // Increment the appropriate isolation sum
      if (thisCandidateType == reco::PFCandidate::h) {
        // for charged hadrons, additionally check consistency
        // with the PV
        float dxy = -999;
        float dz = -999;

        getImpactParameters(CachingPtrCandidate(&*iCand, isAOD_), pv, dxy, dz);

        if (fabs(dxy) > dxyMax || fabs(dz) > dzMax)
          continue;

        // The candidate is eligible, increment the isolaiton
        chargedIsoSum += iCand->pt();
      }

      if (thisCandidateType == reco::PFCandidate::h0)
        neutralHadronIsoSum += iCand->pt();

      if (thisCandidateType == reco::PFCandidate::gamma)
        photonIsoSum += iCand->pt();
    }

    vars[10].push_back(chargedIsoSum);
    vars[11].push_back(neutralHadronIsoSum);
    vars[12].push_back(photonIsoSum);

    // Worst isolation computed with no vetos or ptMin cut, as in Run 1 Hgg code.
    unsigned char options = 0;
    vars[13].push_back(computeWorstPFChargedIsolation(*iPho, pfCandNoNaN, *vertices, pv, options, isAOD_));

    // Worst isolation computed with cone vetos and a ptMin cut, as in Run 2 Hgg code.
    options |= PT_MIN_THRESH | DR_VETO;
    vars[14].push_back(computeWorstPFChargedIsolation(*iPho, pfCandNoNaN, *vertices, pv, options, isAOD_));

    // Like before, but adding primary vertex constraint
    options |= PV_CONSTRAINT;
    vars[15].push_back(computeWorstPFChargedIsolation(*iPho, pfCandNoNaN, *vertices, pv, options, isAOD_));

    // PFCluster Isolations
    vars[16].push_back(iPho->trkSumPtSolidConeDR04());
    if (isAOD_) {
      vars[17].push_back(0.f);
      vars[18].push_back(0.f);
    } else {
      edm::Ptr<pat::Photon> patPhotonPtr{iPho};
      vars[17].push_back(patPhotonPtr->hcalPFClusterIso());
      vars[18].push_back(patPhotonPtr->ecalPFClusterIso());
    }
  }

  // write the value maps
  for (int i = 0; i < nVars_; ++i) {
    auto valMap = std::make_unique<edm::ValueMap<float>>();
    typename edm::ValueMap<float>::Filler filler(*valMap);
    filler.insert(src, vars[i].begin(), vars[i].end());
    filler.fill();
    iEvent.put(std::move(valMap), names[i]);
  }
}

void PhotonIDValueMapProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // photonIDValueMapProducer
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("particleBasedIsolation", edm::InputTag("particleBasedIsolation", "gedPhotons"));
  desc.add<edm::InputTag>("src", edm::InputTag("slimmedPhotons", "", "@skipCurrentProcess"));
  desc.add<edm::InputTag>("esReducedRecHitCollection", edm::InputTag("reducedEgamma", "reducedESRecHits"));
  desc.add<edm::InputTag>("ebReducedRecHitCollection", edm::InputTag("reducedEgamma", "reducedEBRecHits"));
  desc.add<edm::InputTag>("eeReducedRecHitCollection", edm::InputTag("reducedEgamma", "reducedEERecHits"));
  desc.add<edm::InputTag>("pfCandidates", edm::InputTag("packedPFCandidates"));
  desc.add<edm::InputTag>("vertices", edm::InputTag("offlineSlimmedPrimaryVertices"));
  desc.add<bool>("isAOD", false);
  descriptions.add("photonIDValueMapProducer", desc);
}

// Charged isolation with respect to the worst vertex. See more
// comments above at the function declaration.
float PhotonIDValueMapProducer::computeWorstPFChargedIsolation(const reco::Photon& photon,
                                                               const std::vector<edm::Ptr<reco::Candidate>>& pfCands,
                                                               const reco::VertexCollection& vertices,
                                                               const reco::Vertex& pv,
                                                               unsigned char options,
                                                               bool isAOD) const {
  float worstIsolation = 0.0;

  const float dRveto2 = photon.isEB() ? dRveto2Barrel : dRveto2Endcap;

  std::vector<CachingPtrCandidate> chargedCands;
  chargedCands.reserve(pfCands.size());
  for (auto const& aCand : pfCands) {
    // require that PFCandidate is a charged hadron
    reco::PFCandidate::ParticleType thisCandidateType = getCandidatePdgId(&*aCand, isAOD);
    if (thisCandidateType != reco::PFCandidate::h)
      continue;

    if ((options & PT_MIN_THRESH) && aCand.get()->pt() < ptMin)
      continue;

    chargedCands.emplace_back(&*aCand, isAOD);
  }

  // Calculate isolation sum separately for each vertex
  for (unsigned int ivtx = 0; ivtx < vertices.size(); ++ivtx) {
    // Shift the photon according to the vertex
    const reco::VertexRef vtx(&vertices, ivtx);
    math::XYZVector phoWrtVtx(photon.superCluster()->x() - vtx->x(),
                              photon.superCluster()->y() - vtx->y(),
                              photon.superCluster()->z() - vtx->z());

    const float phoWrtVtxPhi = phoWrtVtx.phi();
    const float phoWrtVtxEta = phoWrtVtx.eta();

    float sum = 0;
    // Loop over the PFCandidates
    for (auto const& aCCand : chargedCands) {
      auto iCand = aCCand.candidate;

      float dR2 = deltaR2(phoWrtVtxEta, phoWrtVtxPhi, iCand->eta(), iCand->phi());
      if (dR2 > coneSizeDR2 || (options & DR_VETO && dR2 < dRveto2))
        continue;

      float dxy = -999;
      float dz = -999;
      if (options & PV_CONSTRAINT)
        getImpactParameters(aCCand, pv, dxy, dz);
      else
        getImpactParameters(aCCand, *vtx, dxy, dz);

      if (fabs(dxy) > dxyMax || fabs(dz) > dzMax)
        continue;

      sum += iCand->pt();
    }

    worstIsolation = std::max(sum, worstIsolation);
  }

  return worstIsolation;
}

DEFINE_FWK_MODULE(PhotonIDValueMapProducer);
