// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
//
// class declaration
//

class HiBadParticleCleaner : public edm::global::EDProducer<> {
public:
  explicit HiBadParticleCleaner(const edm::ParameterSet&);
  ~HiBadParticleCleaner() override = default;

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  // ----------member data ---------------------------

  edm::EDGetTokenT<edm::View<reco::PFCandidate>> tokenPFCandidates_;
  edm::EDGetTokenT<reco::VertexCollection> tokenPV_;

  const double minMuonPt_;
  const double minChargedHadronPt_;
  const double minMuonTrackRelPtErr_;
  const double maxSigLoose_;
  const double maxSigTight_;
  const double minCaloCompatibility_;
  const unsigned minTrackNHits_;
  const unsigned minPixelNHits_;
  const int minTrackerLayersForMuonLoose_;
  const int minTrackerLayersForMuonTight_;
};

//
// constructors and destructor
//
HiBadParticleCleaner::HiBadParticleCleaner(const edm::ParameterSet& iConfig)
    : tokenPFCandidates_(consumes<edm::View<reco::PFCandidate>>(iConfig.getParameter<edm::InputTag>("PFCandidates"))),
      tokenPV_(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("offlinePV"))),
      minMuonPt_(iConfig.getParameter<double>("minMuonPt")),
      minChargedHadronPt_(iConfig.getParameter<double>("minChargedHadronPt")),
      minMuonTrackRelPtErr_(iConfig.getParameter<double>("minMuonTrackRelPtErr")),
      maxSigLoose_(iConfig.getParameter<double>("maxSigLoose")),
      maxSigTight_(iConfig.getParameter<double>("maxSigTight")),
      minCaloCompatibility_(iConfig.getParameter<double>("minCaloCompatibility")),
      minTrackNHits_(iConfig.getParameter<uint>("minTrackNHits")),
      minPixelNHits_(iConfig.getParameter<uint>("minPixelNHits")),
      minTrackerLayersForMuonLoose_(iConfig.getParameter<int>("minTrackerLayersForMuonLoose")),
      minTrackerLayersForMuonTight_(iConfig.getParameter<int>("minTrackerLayersForMuonTight")) {
  produces<bool>();
  produces<reco::PFCandidateCollection>();
  produces<reco::PFCandidateCollection>("removed");
  produces<edm::ValueMap<reco::PFCandidateRef>>();
}

//
// member functions
//

// ------------ method called on each new Event  ------------
void HiBadParticleCleaner::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup&) const {
  using namespace std;
  using namespace edm;

  typedef View<reco::PFCandidate> CandidateView;
  Handle<CandidateView> pfCandidates;
  iEvent.getByToken(tokenPFCandidates_, pfCandidates);

  const reco::VertexCollection* recoVertices;
  edm::Handle<reco::VertexCollection> vertexCollection;
  iEvent.getByToken(tokenPV_, vertexCollection);
  recoVertices = vertexCollection.product();

  auto pOutputCandidateCollection = std::make_unique<reco::PFCandidateCollection>();
  auto pBadCandidateCollection = std::make_unique<reco::PFCandidateCollection>();

  bool foundBadCandidate = false;

  size_t n = pfCandidates->size();
  std::vector<int> candidateIndexMapper(n, 0);  // mapping between the original PF and post-cleaning PF
  size_t iPF;
  for (iPF = 0; iPF < n; iPF++) {
    const reco::PFCandidate& pfCandidate = pfCandidates->at(iPF);
    if (pfCandidate.particleId() == reco::PFCandidate::ParticleType::mu)  // muon cleaning
    {
      if (pfCandidate.pt() > minMuonPt_) {
        if (!pfCandidate.muonRef()->isGlobalMuon() || !pfCandidate.muonRef()->isTrackerMuon() ||
            !pfCandidate.trackRef().isNonnull()) {
          foundBadCandidate = true;
          continue;
        }
        reco::TrackRef track = pfCandidate.trackRef();

        if (track->ptError() / track->pt() > minMuonTrackRelPtErr_ || track->pt() < pfCandidate.pt() / 2.) {
          foundBadCandidate = true;
          continue;
        }

        if (track->algo() == reco::TrackBase::muonSeededStepInOut ||
            track->algo() == reco::TrackBase::muonSeededStepOutIn ||
            track->originalAlgo() == reco::TrackBase::muonSeededStepInOut ||
            track->originalAlgo() == reco::TrackBase::muonSeededStepOutIn ||
            track->hitPattern().trackerLayersWithMeasurement() < minTrackerLayersForMuonLoose_) {
          const reco::Vertex& vtx = (*recoVertices)[0];
          float bestVzError = vtx.zError();
          const math::XYZPoint& bestVtx(vtx.position());
          math::Error<3>::type vtx_cov = vtx.covariance();
          float dz = std::abs(track->dz(bestVtx));
          float dxy = std::abs(track->dxy(bestVtx));
          float dzError2 = track->dzError() * track->dzError() + bestVzError * bestVzError;
          float dxyError = track->dxyError(bestVtx, vtx_cov);

          float dzSig2 = dz * dz / dzError2;
          float dxySig2 = dxy * dxy / dxyError / dxyError;

          float sig3d = sqrt(dzSig2 + dxySig2);

          if (sig3d > maxSigLoose_) {
            pBadCandidateCollection->push_back(pfCandidate);
            candidateIndexMapper[iPF] = -1 * (pBadCandidateCollection->size());
            foundBadCandidate = true;
            continue;
          }

          if (track->pt() < pfCandidate.pt() / 1.5 || track->pt() > pfCandidate.pt() * 1.5) {
            foundBadCandidate = true;
            pBadCandidateCollection->push_back(pfCandidate);
            candidateIndexMapper[iPF] = -1 * (pBadCandidateCollection->size());
            continue;
          }
          if (track->originalAlgo() == reco::TrackBase::muonSeededStepOutIn &&
              track->hitPattern().trackerLayersWithMeasurement() < minTrackerLayersForMuonTight_) {
            foundBadCandidate = true;
            pBadCandidateCollection->push_back(pfCandidate);
            candidateIndexMapper[iPF] = -1 * (pBadCandidateCollection->size());
            continue;
          }
        }
      }
    } else if (pfCandidate.particleId() == reco::PFCandidate::ParticleType::h)  //charged hadron cleaning
    {
      if (pfCandidate.pt() > minChargedHadronPt_) {
        reco::TrackRef track = pfCandidate.trackRef();

        unsigned nHits = track->numberOfValidHits();
        unsigned nPixelHits = track->hitPattern().numberOfValidPixelHits();

        if ((nHits < minTrackNHits_ && nPixelHits < minPixelNHits_) || nHits == 3) {
          foundBadCandidate = true;
          pBadCandidateCollection->push_back(pfCandidate);
          candidateIndexMapper[iPF] = -1 * (pBadCandidateCollection->size());
          continue;
        }

        const reco::Vertex& vtx = (*recoVertices)[0];
        float bestVzError = vtx.zError();
        const math::XYZPoint& bestVtx(vtx.position());
        math::Error<3>::type vtx_cov = vtx.covariance();
        float dz = std::abs(track->dz(bestVtx));
        float dxy = std::abs(track->dxy(bestVtx));
        float dzError2 = track->dzError() * track->dzError() + bestVzError * bestVzError;
        float dxyError = track->dxyError(bestVtx, vtx_cov);

        float dzSig2 = dz * dz / dzError2;
        float dxySig2 = dxy * dxy / dxyError / dxyError;

        float sig3d = sqrt(dzSig2 + dxySig2);

        if (sig3d > maxSigLoose_) {
          foundBadCandidate = true;
          pBadCandidateCollection->push_back(pfCandidate);
          candidateIndexMapper[iPF] = -1 * (pBadCandidateCollection->size());
          continue;
        }

        if (sig3d > maxSigTight_ && nHits < minTrackNHits_) {
          foundBadCandidate = true;
          pBadCandidateCollection->push_back(pfCandidate);
          candidateIndexMapper[iPF] = -1 * (pBadCandidateCollection->size());
          continue;
        }

        if (track->algo() == reco::TrackBase::muonSeededStepInOut ||
            track->algo() == reco::TrackBase::muonSeededStepOutIn ||
            track->originalAlgo() == reco::TrackBase::muonSeededStepInOut ||
            track->originalAlgo() == reco::TrackBase::muonSeededStepOutIn) {
          if (sig3d > maxSigLoose_) {
            foundBadCandidate = true;
            pBadCandidateCollection->push_back(pfCandidate);
            candidateIndexMapper[iPF] = -1 * (pBadCandidateCollection->size());
            continue;
          }

          if (nHits < minTrackNHits_) {
            foundBadCandidate = true;
            pBadCandidateCollection->push_back(pfCandidate);
            candidateIndexMapper[iPF] = -1 * (pBadCandidateCollection->size());
            continue;
          }
        }

        double caloEnergy = pfCandidate.ecalEnergy() + pfCandidate.hcalEnergy();

        if (caloEnergy < track->p() * minCaloCompatibility_) {
          if (sig3d > maxSigTight_) {
            foundBadCandidate = true;
            pBadCandidateCollection->push_back(pfCandidate);
            candidateIndexMapper[iPF] = -1 * (pBadCandidateCollection->size());
            continue;
          }

          if (nHits < minTrackNHits_) {
            foundBadCandidate = true;
            pBadCandidateCollection->push_back(pfCandidate);
            candidateIndexMapper[iPF] = -1 * (pBadCandidateCollection->size());
            continue;
          }

          if (nPixelHits < minPixelNHits_) {
            foundBadCandidate = true;
            pBadCandidateCollection->push_back(pfCandidate);
            candidateIndexMapper[iPF] = -1 * (pBadCandidateCollection->size());
            continue;
          }
        }
      }
    }

    pOutputCandidateCollection->push_back(pfCandidate);
    candidateIndexMapper[iPF] = (pOutputCandidateCollection->size());
  }  // end loop over pf candidates

  bool pass = !foundBadCandidate;

  edm::OrphanHandle<std::vector<reco::PFCandidate>> newpf = iEvent.put(std::move(pOutputCandidateCollection));
  edm::OrphanHandle<std::vector<reco::PFCandidate>> badpf = iEvent.put(std::move(pBadCandidateCollection), "removed");

  iEvent.put(std::make_unique<bool>(pass));

  std::unique_ptr<edm::ValueMap<reco::PFCandidateRef>> pf2pf(new edm::ValueMap<reco::PFCandidateRef>());
  edm::ValueMap<reco::PFCandidateRef>::Filler filler(*pf2pf);

  std::vector<reco::PFCandidateRef> refs;
  refs.reserve(n);

  for (iPF = 0; iPF < n; ++iPF) {
    if (candidateIndexMapper[iPF] > 0) {
      refs.push_back(reco::PFCandidateRef(newpf, candidateIndexMapper[iPF] - 1));
    } else if (candidateIndexMapper[iPF] < 0) {
      refs.push_back(reco::PFCandidateRef(badpf, -candidateIndexMapper[iPF] - 1));
    }
  }
  filler.insert(pfCandidates, refs.begin(), refs.end());

  filler.fill();
  iEvent.put(std::move(pf2pf));
}

//define this as a plug-in
DEFINE_FWK_MODULE(HiBadParticleCleaner);
