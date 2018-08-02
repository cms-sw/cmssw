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
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include <FWCore/ParameterSet/interface/ConfigurationDescriptions.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <FWCore/ParameterSet/interface/ParameterSetDescription.h>
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"

// This template function finds whether theCandidate is in thefootprint
// collection. It is templated to be able to handle both reco and pat
// photons (from AOD and miniAOD, respectively).
template <class T, class U>
bool isInFootprint(const T& footprint, const U& candidate)
{
    for (auto& it : footprint) {
        if (it.key() == candidate.key())
            return true;
    }
    return false;
}

class PhotonIDValueMapProducer : public edm::stream::EDProducer<> {

public:
    explicit PhotonIDValueMapProducer(const edm::ParameterSet&);
    ~PhotonIDValueMapProducer() override;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
    void produce(edm::Event&, const edm::EventSetup&) override;

    // This function computes charged hadron isolation with respect to multiple
    // PVs and returns the worst of the found isolation values. The function
    // implements the computation method taken directly from Run 1 code of
    // H->gamma gamma, specifically from the class CiCPhotonID of the
    // HiggsTo2photons anaysis code. Template is introduced to handle reco/pat
    // photons and aod/miniAOD PF candidates collections
    template <class T, class U>
    float computeWorstPFChargedIsolation(
            const T& photon,
            const U& pfCands,
            const edm::Handle<reco::VertexCollection> vertices,
            const reco::Vertex& pv,
            unsigned char options);

    // Some helper functions that are needed to access info in
    // AOD vs miniAOD
    reco::PFCandidate::ParticleType candidatePdgId(const edm::Ptr<reco::Candidate> candidate);

    const reco::Track* getTrackPointer(const edm::Ptr<reco::Candidate> candidate);
    void getImpactParameters(
        const edm::Ptr<reco::Candidate>& candidate, const reco::Vertex& pv, float& dxy, float& dz);

    // The object that will compute 5x5 quantities
    std::unique_ptr<noZS::EcalClusterLazyTools> lazyToolnoZS;

    // check whether a non-null preshower is there
    const bool usesES_;

    // for AOD case
    const edm::EDGetTokenT<EcalRecHitCollection> ebReducedRecHitCollection_;
    const edm::EDGetTokenT<EcalRecHitCollection> eeReducedRecHitCollection_;
    const edm::EDGetTokenT<EcalRecHitCollection> esReducedRecHitCollection_;
    const edm::EDGetTokenT<reco::VertexCollection> vtxToken_;
    const edm::EDGetTokenT<edm::ValueMap<std::vector<reco::PFCandidateRef>>> particleBasedIsolationToken_;
    const edm::EDGetTokenT<edm::View<reco::Candidate>> pfCandsToken_;
    const edm::EDGetToken src_;

    // for miniAOD case
    const edm::EDGetTokenT<EcalRecHitCollection> ebReducedRecHitCollectionMiniAOD_;
    const edm::EDGetTokenT<EcalRecHitCollection> eeReducedRecHitCollectionMiniAOD_;
    const edm::EDGetTokenT<EcalRecHitCollection> esReducedRecHitCollectionMiniAOD_;
    const edm::EDGetTokenT<reco::VertexCollection> vtxTokenMiniAOD_;
    const edm::EDGetTokenT<edm::ValueMap<std::vector<reco::PFCandidateRef>>> particleBasedIsolationTokenMiniAOD_;
    const edm::EDGetTokenT<edm::View<reco::Candidate>> pfCandsTokenMiniAOD_;
    const edm::EDGetToken srcMiniAOD_;

    bool isAOD_;
};

constexpr int nVars_ = 19;

const std::string names[nVars_] = {
    // Cluster shapes
    "phoFull5x5SigmaIEtaIEta", // 0
    "phoFull5x5SigmaIEtaIPhi",
    "phoFull5x5E1x3",
    "phoFull5x5E2x2",
    "phoFull5x5E2x5Max",
    "phoFull5x5E5x5", // 5
    "phoESEffSigmaRR",
    // Cluster shape ratios
    "phoFull5x5E1x3byE5x5",
    "phoFull5x5E2x2byE5x5",
    "phoFull5x5E2x5byE5x5",
    // Isolations
    "phoChargedIsolation", // 10
    "phoNeutralHadronIsolation",
    "phoPhotonIsolation",
    "phoWorstChargedIsolation",
    "phoWorstChargedIsolationConeVeto",
    "phoWorstChargedIsolationConeVetoPVConstr", // 15
    // PFCluster Isolation
    "phoTrkIsolation",
    "phoHcalPFClIsolation",
    "phoEcalPFClIsolation"
};

// options and bitflags
constexpr float coneSizeDR = 0.3;
constexpr float dxyMax = 0.1;
constexpr float dzMax = 0.2;
constexpr float dRvetoBarrel = 0.02;
constexpr float dRvetoEndcap = 0.02;
constexpr float ptMin = 0.1;

const unsigned char PV_CONSTRAINT  = 0x1;
const unsigned char DR_VETO = 0x2;
const unsigned char PT_MIN_THRESH = 0x8;

PhotonIDValueMapProducer::PhotonIDValueMapProducer(const edm::ParameterSet& cfg)
    : usesES_(!cfg.getParameter<edm::InputTag>("esReducedRecHitCollection").label().empty()
          || !cfg.getParameter<edm::InputTag>("esReducedRecHitCollectionMiniAOD").label().empty())
    // Declare consummables, handle both AOD and miniAOD case
    , ebReducedRecHitCollection_(
          mayConsume<EcalRecHitCollection>(cfg.getParameter<edm::InputTag>("ebReducedRecHitCollection")))
    , eeReducedRecHitCollection_(
          mayConsume<EcalRecHitCollection>(cfg.getParameter<edm::InputTag>("eeReducedRecHitCollection")))
    , esReducedRecHitCollection_(
          mayConsume<EcalRecHitCollection>(cfg.getParameter<edm::InputTag>("esReducedRecHitCollection")))
    , vtxToken_(mayConsume<reco::VertexCollection>(cfg.getParameter<edm::InputTag>("vertices")))
    // The particleBasedIsolation object is relevant only for AOD, RECO format
    , particleBasedIsolationToken_(mayConsume<edm::ValueMap<std::vector<reco::PFCandidateRef>>>(
          cfg.getParameter<edm::InputTag>("particleBasedIsolation")))
    // AOD has reco::PFCandidate vector, and miniAOD has pat::PackedCandidate
    // vector. Both inherit from reco::Candidate, so we go to the base class.
    // We can cast into the full type later if needed. Since the collection
    // names are different, we introduce both collections
    , pfCandsToken_(mayConsume<edm::View<reco::Candidate>>(cfg.getParameter<edm::InputTag>("pfCandidates")))
    // reco photons are castable into pat photons, so no need to handle reco/pat seprately
    , src_(mayConsume<edm::View<reco::Photon>>(cfg.getParameter<edm::InputTag>("src")))
    , ebReducedRecHitCollectionMiniAOD_(
          mayConsume<EcalRecHitCollection>(cfg.getParameter<edm::InputTag>("ebReducedRecHitCollectionMiniAOD")))
    , eeReducedRecHitCollectionMiniAOD_(
          mayConsume<EcalRecHitCollection>(cfg.getParameter<edm::InputTag>("eeReducedRecHitCollectionMiniAOD")))
    , esReducedRecHitCollectionMiniAOD_(
          mayConsume<EcalRecHitCollection>(cfg.getParameter<edm::InputTag>("esReducedRecHitCollectionMiniAOD")))
    , vtxTokenMiniAOD_(mayConsume<reco::VertexCollection>(cfg.getParameter<edm::InputTag>("verticesMiniAOD")))
    , pfCandsTokenMiniAOD_(
          mayConsume<edm::View<reco::Candidate>>(cfg.getParameter<edm::InputTag>("pfCandidatesMiniAOD")))
    , srcMiniAOD_(mayConsume<edm::View<reco::Photon>>(cfg.getParameter<edm::InputTag>("srcMiniAOD")))
{

    // Declare producibles
    for (int i = 0; i < nVars_; ++i)
        produces<edm::ValueMap<float>>(names[i]);
}

PhotonIDValueMapProducer::~PhotonIDValueMapProducer() {}

void PhotonIDValueMapProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    edm::Handle<edm::View<reco::Photon>> src;

    isAOD_ = true;
    iEvent.getByToken(src_, src);
    if (!src.isValid()) {
        isAOD_ = false;
        iEvent.getByToken(srcMiniAOD_, src);
    }
    if (!src.isValid()) {
        throw cms::Exception("IllDefinedDataTier") << "DataFormat does not contain a photon source!";
    }

    // Configure Lazy Tools
    if (usesES_) {
        if (isAOD_)
            lazyToolnoZS = std::make_unique<noZS::EcalClusterLazyTools>(
                iEvent, iSetup, ebReducedRecHitCollection_, eeReducedRecHitCollection_, esReducedRecHitCollection_);
        else
            lazyToolnoZS
                = std::make_unique<noZS::EcalClusterLazyTools>(iEvent, iSetup, ebReducedRecHitCollectionMiniAOD_,
                    eeReducedRecHitCollectionMiniAOD_, esReducedRecHitCollectionMiniAOD_);
    } else {
        if (isAOD_)
            lazyToolnoZS = std::make_unique<noZS::EcalClusterLazyTools>(
                iEvent, iSetup, ebReducedRecHitCollection_, eeReducedRecHitCollection_);
        else
            lazyToolnoZS = std::make_unique<noZS::EcalClusterLazyTools>(
                iEvent, iSetup, ebReducedRecHitCollectionMiniAOD_, eeReducedRecHitCollectionMiniAOD_);
    }

    // Get PV
    edm::Handle<reco::VertexCollection> vertices;
    iEvent.getByToken(vtxToken_, vertices);
    if (!vertices.isValid())
        iEvent.getByToken(vtxTokenMiniAOD_, vertices);
    if (vertices->empty())
        return; // skip the event if no PV found
    const reco::Vertex& pv = vertices->front();

    edm::Handle<edm::ValueMap<std::vector<reco::PFCandidateRef>>> particleBasedIsolationMap;
    if (isAOD_) {
        // this exists only in AOD
        iEvent.getByToken(particleBasedIsolationToken_, particleBasedIsolationMap);
    }

    edm::Handle<edm::View<reco::Candidate>> pfCandsHandle;

    iEvent.getByToken(pfCandsToken_, pfCandsHandle);
    if (!pfCandsHandle.isValid())
        iEvent.getByToken(pfCandsTokenMiniAOD_, pfCandsHandle);

    if (!isAOD_ && !src->empty()) {
        edm::Ptr<pat::Photon> test(src->ptrAt(0));
        if (test.isNull() || !test.isAvailable()) {
            throw cms::Exception("InvalidConfiguration")
                << "DataFormat is detected as miniAOD but cannot cast to pat::Photon!";
        }
    }

    std::vector<float> vars[nVars_];

    // reco::Photon::superCluster() is virtual so we can exploit polymorphism
    for (unsigned i = 0; i < src->size(); ++i) {
        const auto& iPho = src->ptrAt(i);

        //
        // Compute full 5x5 quantities
        //
        const auto& theseed = *(iPho->superCluster()->seed());

        // For full5x5_sigmaIetaIeta, for 720 we use: lazy tools for AOD,
        // and userFloats or lazy tools for miniAOD. From some point in 72X and on, one can
        // retrieve the full5x5 directly from the object with ->full5x5_sigmaIetaIeta()
        // for both formats.
        std::vector<float> vCov = lazyToolnoZS->localCovariances(theseed);
        vars[0].push_back(isnan(vCov[0]) ? 0. : sqrt(vCov[0]));
        vars[1].push_back(vCov[1]);
        vars[2].push_back(lazyToolnoZS->e1x3(theseed));
        vars[3].push_back(lazyToolnoZS->e2x2(theseed));
        vars[4].push_back(lazyToolnoZS->e2x5Max(theseed));
        vars[5].push_back(lazyToolnoZS->e5x5(theseed));
        vars[6].push_back(lazyToolnoZS->eseffsirir(*(iPho->superCluster())));
        vars[7].push_back(vars[2][i] / vars[5][i]);
        vars[8].push_back(vars[3][i] / vars[5][i]);
        vars[9].push_back(vars[4][i] / vars[5][i]);

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

        // Loop over all PF candidates
        for (unsigned int idxcand = 0; idxcand < pfCandsHandle->size(); ++idxcand) {

            // Here, the type will be a simple reco::Candidate. We cast it
            // for full PFCandidate or PackedCandidate below as necessary
            const auto& iCand = pfCandsHandle->ptrAt(idxcand);

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
            if (dR2 > coneSizeDR * coneSizeDR)
                continue;

            // Check if this candidate is not in the footprint
            if (isAOD_) {
                if(isInFootprint((*particleBasedIsolationMap)[iPho], iCand))
                    continue;
            } else {
                edm::Ptr<pat::Photon> patPhotonPtr(src->ptrAt(i));
                if(isInFootprint(patPhotonPtr->associatedPackedPFCandidates(), iCand))
                    continue;
            }

            // Find candidate type
            reco::PFCandidate::ParticleType thisCandidateType = candidatePdgId(iCand);

            // Increment the appropriate isolation sum
            if (thisCandidateType == reco::PFCandidate::h) {
                // for charged hadrons, additionally check consistency
                // with the PV
                float dxy = -999;
                float dz = -999;
                getImpactParameters(iCand, pv, dxy, dz);

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
        vars[13].push_back(computeWorstPFChargedIsolation(iPho, pfCandsHandle, vertices, pv, options));

        // Worst isolation computed with cone vetos and a ptMin cut, as in Run 2 Hgg code.
        options |= PT_MIN_THRESH | DR_VETO;
        vars[14].push_back(computeWorstPFChargedIsolation(iPho, pfCandsHandle, vertices, pv, options));

        // Like before, but adding primary vertex constraint
        options |= PV_CONSTRAINT;
        vars[15].push_back(computeWorstPFChargedIsolation(iPho, pfCandsHandle, vertices, pv, options));

        // PFCluster Isolations
        vars[16].push_back(iPho->trkSumPtSolidConeDR04());
        if (isAOD_) {
            vars[17].push_back(0.f);
            vars[18].push_back(0.f);
        } else {
            edm::Ptr<pat::Photon> patPhotonPtr{ src->ptrAt(i) };
            vars[17].push_back(patPhotonPtr->hcalPFClusterIso());
            vars[18].push_back(patPhotonPtr->ecalPFClusterIso());
        }

    }

    // write the value maps
    for (int i = 0; i < nVars_; ++i) {
        auto valMap = std::make_unique<edm::ValueMap<float>>();
        edm::ValueMap<float>::Filler filler(*valMap);
        filler.insert(src, vars[i].begin(), vars[i].end());
        filler.fill();
        iEvent.put(std::move(valMap), names[i]);
    }
}

void PhotonIDValueMapProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // photonIDValueMapProducer
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("particleBasedIsolation",           edm::InputTag("particleBasedIsolation","gedPhotons"));
  desc.add<edm::InputTag>("src",                              edm::InputTag("gedPhotons"));
  desc.add<edm::InputTag>("srcMiniAOD",                       edm::InputTag("slimmedPhotons","","@skipCurrentProcess"));
  desc.add<edm::InputTag>("esReducedRecHitCollectionMiniAOD", edm::InputTag("reducedEgamma","reducedESRecHits"));
  desc.add<edm::InputTag>("eeReducedRecHitCollection",        edm::InputTag("reducedEcalRecHitsEE"));
  desc.add<edm::InputTag>("pfCandidates",                     edm::InputTag("particleFlow"));
  desc.add<edm::InputTag>("vertices",                         edm::InputTag("offlinePrimaryVertices"));
  desc.add<edm::InputTag>("ebReducedRecHitCollectionMiniAOD", edm::InputTag("reducedEgamma","reducedEBRecHits"));
  desc.add<edm::InputTag>("eeReducedRecHitCollectionMiniAOD", edm::InputTag("reducedEgamma","reducedEERecHits"));
  desc.add<edm::InputTag>("esReducedRecHitCollection",        edm::InputTag("reducedEcalRecHitsES"));
  desc.add<edm::InputTag>("pfCandidatesMiniAOD",              edm::InputTag("packedPFCandidates"));
  desc.add<edm::InputTag>("verticesMiniAOD",                  edm::InputTag("offlineSlimmedPrimaryVertices"));
  desc.add<edm::InputTag>("ebReducedRecHitCollection",        edm::InputTag("reducedEcalRecHitsEB"));
  descriptions.add("photonIDValueMapProducer", desc);
}


// Charged isolation with respect to the worst vertex. See more
// comments above at the function declaration.
template <class T, class U>
float PhotonIDValueMapProducer ::computeWorstPFChargedIsolation(const T& photon, const U& pfCands,
    const edm::Handle<reco::VertexCollection> vertices, const reco::Vertex& pv, unsigned char options)
{
    float worstIsolation = 0.0;

    const float dRveto = photon->isEB() ? dRvetoBarrel : dRvetoEndcap;

    // Calculate isolation sum separately for each vertex
    for (unsigned int ivtx = 0; ivtx < vertices->size(); ++ivtx) {

        // Shift the photon according to the vertex
        reco::VertexRef vtx(vertices, ivtx);
        math::XYZVector phoWrtVtx(photon->superCluster()->x() - vtx->x(),
            photon->superCluster()->y() - vtx->y(), photon->superCluster()->z() - vtx->z());

        float sum = 0;
        // Loop over the PFCandidates
        for (unsigned int i = 0; i < pfCands->size(); i++) {

            const auto& iCand = pfCands->ptrAt(i);

            // require that PFCandidate is a charged hadron
            reco::PFCandidate::ParticleType thisCandidateType = candidatePdgId(iCand);
            if (thisCandidateType != reco::PFCandidate::h)
                continue;

            if ((options & PT_MIN_THRESH) && iCand->pt() < ptMin)
                continue;

            float dxy = -999;
            float dz = -999;
            if (options & PV_CONSTRAINT)
                getImpactParameters(iCand, pv, dxy, dz);
            else
                getImpactParameters(iCand, *vtx, dxy, dz);

            if (fabs(dxy) > dxyMax || fabs(dz) > dzMax)
                continue;

            float dR2 = deltaR2(phoWrtVtx.Eta(), phoWrtVtx.Phi(), iCand->eta(), iCand->phi());
            if (dR2 > coneSizeDR * coneSizeDR ||
                    (options & DR_VETO && dR2 < dRveto * dRveto))
                continue;

            sum += iCand->pt();
        }

        worstIsolation = std::max(sum, worstIsolation);
    }

    return worstIsolation;
}

reco::PFCandidate::ParticleType PhotonIDValueMapProducer::candidatePdgId(
    const edm::Ptr<reco::Candidate> candidate)
{
    if (isAOD_)
        return ((const edm::Ptr<reco::PFCandidate>)candidate)->particleId();

    // the neutral hadrons and charged hadrons can be of pdgId types
    // only 130 (K0L) and +-211 (pi+-) in packed candidates
    const int pdgId = ((const edm::Ptr<pat::PackedCandidate>)candidate)->pdgId();
    if (pdgId == 22)
        return reco::PFCandidate::gamma;
    else if (abs(pdgId) == 130) // PDG ID for K0L
        return reco::PFCandidate::h0;
    else if (abs(pdgId) == 211) // PDG ID for pi+-
        return reco::PFCandidate::h;
    else
        return reco::PFCandidate::X;
}

const reco::Track* PhotonIDValueMapProducer::getTrackPointer(const edm::Ptr<reco::Candidate> candidate)
{
    return isAOD_ ?
        &*(((const edm::Ptr<reco::PFCandidate>)candidate)->trackRef()) :
        &(((const edm::Ptr<pat::PackedCandidate>)candidate)->pseudoTrack());
}

void PhotonIDValueMapProducer::getImpactParameters(
    const edm::Ptr<reco::Candidate>& candidate, const reco::Vertex& pv, float& dxy, float& dz)
{
    if (isAOD_) {
        auto theTrack = *static_cast<const edm::Ptr<reco::PFCandidate>>(candidate)->trackRef();
        dxy = theTrack.dxy(pv.position());
        dz = theTrack.dz(pv.position());
    } else {
        auto aCand = *static_cast<const edm::Ptr<pat::PackedCandidate>>(candidate);
        dxy = aCand.dxy(pv.position());
        dz = aCand.dz(pv.position());
    }
}

DEFINE_FWK_MODULE(PhotonIDValueMapProducer);
