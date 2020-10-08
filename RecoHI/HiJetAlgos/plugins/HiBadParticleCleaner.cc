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

  edm::EDGetTokenT<edm::View<reco::PFCandidate> > tokenPFCandidates_;
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
    : tokenPFCandidates_(consumes<edm::View<reco::PFCandidate> >(iConfig.getParameter<edm::InputTag>("PFCandidates"))),
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

  for (const reco::PFCandidate& pfCandidate : *pfCandidates) {
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
            foundBadCandidate = true;
            continue;
          }

          if (track->pt() < pfCandidate.pt() / 1.5 || track->pt() > pfCandidate.pt() * 1.5) {
            foundBadCandidate = true;
            pBadCandidateCollection->push_back(pfCandidate);
            continue;
          }
          if (track->originalAlgo() == reco::TrackBase::muonSeededStepOutIn &&
              track->hitPattern().trackerLayersWithMeasurement() < minTrackerLayersForMuonTight_) {
            foundBadCandidate = true;
            pBadCandidateCollection->push_back(pfCandidate);
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
          continue;
        }

        if (sig3d > maxSigTight_ && nHits < minTrackNHits_) {
          foundBadCandidate = true;
          pBadCandidateCollection->push_back(pfCandidate);
          continue;
        }

        if (track->algo() == reco::TrackBase::muonSeededStepInOut ||
            track->algo() == reco::TrackBase::muonSeededStepOutIn ||
            track->originalAlgo() == reco::TrackBase::muonSeededStepInOut ||
            track->originalAlgo() == reco::TrackBase::muonSeededStepOutIn) {
          if (sig3d > maxSigLoose_) {
            foundBadCandidate = true;
            pBadCandidateCollection->push_back(pfCandidate);
            continue;
          }

          if (nHits < minTrackNHits_) {
            foundBadCandidate = true;
            pBadCandidateCollection->push_back(pfCandidate);
            continue;
          }
        }

        double caloEnergy = pfCandidate.ecalEnergy() + pfCandidate.hcalEnergy();

        if (caloEnergy < track->p() * minCaloCompatibility_) {
          if (sig3d > maxSigTight_) {
            foundBadCandidate = true;
            pBadCandidateCollection->push_back(pfCandidate);
            continue;
          }

          if (nHits < minTrackNHits_) {
            foundBadCandidate = true;
            pBadCandidateCollection->push_back(pfCandidate);
            continue;
          }

          if (nPixelHits < minPixelNHits_) {
            foundBadCandidate = true;
            pBadCandidateCollection->push_back(pfCandidate);
            continue;
          }
        }
      }
    }

    pOutputCandidateCollection->push_back(pfCandidate);

  }  // end loop over pf candidates

  bool pass = !foundBadCandidate;

  iEvent.put(std::move(pOutputCandidateCollection));
  iEvent.put(std::move(pBadCandidateCollection), "removed");

  iEvent.put(std::make_unique<bool>(pass));
}

//define this as a plug-in
DEFINE_FWK_MODULE(HiBadParticleCleaner);
