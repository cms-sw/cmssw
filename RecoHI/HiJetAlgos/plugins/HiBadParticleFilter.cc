// system include files
#include <memory>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Math/interface/deltaR.h"
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

class HiBadParticleFilter : public edm::stream::EDProducer<> {
public:
  explicit HiBadParticleFilter(const edm::ParameterSet&);
  ~HiBadParticleFilter() override;

private:
  void produce(edm::Event&, const edm::EventSetup&) override;
  // ----------member data ---------------------------

  edm::EDGetTokenT<edm::View<reco::PFCandidate> > tokenPFCandidates_;
  edm::EDGetTokenT<reco::VertexCollection> tokenPV_;

  const bool verbose_;
  const double minMuonPt_;
  const double minChargedHadronPt_;
  const double minMuonTrackRelPtErr_;
  const double maxSigLoose_;
  const double maxSigTight_;
  const double minCaloCompatibility_;
  const unsigned minTrackNHits_;
  const unsigned minPixelNHits_;
};

//
// constructors and destructor
//
HiBadParticleFilter::HiBadParticleFilter(const edm::ParameterSet& iConfig)
    : tokenPFCandidates_(consumes<edm::View<reco::PFCandidate> >(iConfig.getParameter<edm::InputTag>("PFCandidates"))),
      tokenPV_(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("offlinePV"))),
      verbose_(iConfig.getParameter<bool>("verbose")),
      minMuonPt_(iConfig.getParameter<double>("minMuonPt")),
      minChargedHadronPt_(iConfig.getParameter<double>("minChargedHadronPt")),
      minMuonTrackRelPtErr_(iConfig.getParameter<double>("minMuonTrackRelPtErr")),
      maxSigLoose_(iConfig.getParameter<double>("maxSigLoose")),
      maxSigTight_(iConfig.getParameter<double>("maxSigTight")),
      minCaloCompatibility_(iConfig.getParameter<double>("minCaloCompatibility")),
      minTrackNHits_(iConfig.getParameter<uint>("minTrackNHits")),
      minPixelNHits_(iConfig.getParameter<uint>("minPixelNHits")) {
  produces<bool>();
  produces<reco::PFCandidateCollection>();
  produces<reco::PFCandidateCollection>("cleaned");
}

HiBadParticleFilter::~HiBadParticleFilter() {}

//
// member functions
//

// ------------ method called on each new Event  ------------
void HiBadParticleFilter::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
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

  for (unsigned j = 0; j < pfCandidates->size(); ++j) {
    const reco::PFCandidate& pfCandidate = (*pfCandidates)[j];

    if (pfCandidate.particleId() == 3)  // muon cleaning
    {
      if (pfCandidate.pt() > minMuonPt_) {
        if (!pfCandidate.muonRef()->isGlobalMuon() || !pfCandidate.muonRef()->isTrackerMuon() ||
            !pfCandidate.trackRef().isNonnull()) {
          if (verbose_)
            std::cout << " bad muon fit " << pfCandidate.pt() << std::endl;
          cout << " isGlobal " << pfCandidate.muonRef()->isGlobalMuon() << " isTracker "
               << pfCandidate.muonRef()->isTrackerMuon() << " has track " << pfCandidate.trackRef().isNonnull()
               << std::endl;
          foundBadCandidate = true;
          continue;
        }
        reco::TrackRef track = pfCandidate.trackRef();

        if (track->ptError() / track->pt() > minMuonTrackRelPtErr_ || track->pt() < pfCandidate.pt() / 2.) {
          if (verbose_) {
            std::cout << " bad muon err " << pfCandidate.pt() << std::endl;
            std::cout << " rel err = " << track->ptError() / track->pt() << std::endl;
          }
          foundBadCandidate = true;
          continue;
        }

        if (track->algo() == 13 || track->algo() == 14 || track->originalAlgo() == 14 || track->originalAlgo() == 13 ||
            track->hitPattern().trackerLayersWithMeasurement() < 7) {
          const reco::Vertex& vtx = (*recoVertices)[0];
          float bestVzError = vtx.zError();
          const math::XYZPoint& bestVtx(vtx.position());
          math::Error<3>::type vtx_cov = vtx.covariance();
          float dz = std::abs(track->dz(bestVtx));
          float dxy = std::abs(track->dxy(bestVtx));
          float dzError = std::hypot(track->dzError(), bestVzError);
          float dxyError = track->dxyError(bestVtx, vtx_cov);

          float dzSig = dz / dzError;
          float dxySig = dxy / dxyError;

          float sig3d = sqrt(dxySig * dxySig + dzSig * dzSig);

          if (sig3d > maxSigLoose_) {
            if (verbose_) {
              std::cout << " bad muon algo 14, large IP " << pfCandidate.pt() << std::endl;
              std::cout << " dxy " << dxy << " dxy err " << dxyError << std::endl;
              std::cout << " dz " << dz << " dz err " << dzError << std::endl;
            }
            pBadCandidateCollection->push_back(pfCandidate);
            foundBadCandidate = true;
            continue;
          }

          if (track->pt() < pfCandidate.pt() / 1.5 || track->pt() > pfCandidate.pt() * 1.5) {
            if (verbose_) {
              std::cout << " bad muon algo, bad ptack pT " << pfCandidate.pt() << std::endl;
              std::cout << " track pT " << track->pt() << " cand pT " << pfCandidate.pt() << std::endl;
            }
            foundBadCandidate = true;
            pBadCandidateCollection->push_back(pfCandidate);
            continue;
          }
          if (track->originalAlgo() == 14 && track->hitPattern().trackerLayersWithMeasurement() < 10) {
            if (verbose_) {
              std::cout << " bad muon original algo 14, small number of hits " << pfCandidate.pt() << std::endl;
              std::cout << " trakc N hits " << track->hitPattern().trackerLayersWithMeasurement() << std::endl;
            }
            foundBadCandidate = true;
            pBadCandidateCollection->push_back(pfCandidate);
            continue;
          }
        }
      }
    } else if (pfCandidate.particleId() == 1)  //charged hadron cleaning
    {
      if (pfCandidate.pt() > minChargedHadronPt_) {
        reco::TrackRef track = pfCandidate.trackRef();

        unsigned nHits = track->numberOfValidHits();
        unsigned nPixelHits = track->hitPattern().numberOfValidPixelHits();

        if ((nHits < minTrackNHits_ && nPixelHits < minPixelNHits_) || nHits == 3) {
          if (verbose_)
            std::cout << " bad  track with small nPixelHits, pT = " << pfCandidate.pt() << ", nhits = " << nPixelHits
                      << std::endl;
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
        float dzError = std::hypot(track->dzError(), bestVzError);
        float dxyError = track->dxyError(bestVtx, vtx_cov);

        float dzSig = dz / dzError;
        float dxySig = dxy / dxyError;

        float sig3d = sqrt(dxySig * dxySig + dzSig * dzSig);

        if (sig3d > maxSigLoose_) {
          if (verbose_)
            std::cout << " bad  track impact parameter, pT = " << pfCandidate.pt() << ", dxySig = " << dxySig
                      << ", dzSig = " << dzSig << std::endl;
          foundBadCandidate = true;
          pBadCandidateCollection->push_back(pfCandidate);
          continue;
        }

        if (sig3d > maxSigTight_ && nHits < minTrackNHits_) {
          if (verbose_)
            std::cout << " bad  track with small nhits, pT = " << pfCandidate.pt() << ", nhits = " << nHits
                      << std::endl;
          foundBadCandidate = true;
          pBadCandidateCollection->push_back(pfCandidate);
          continue;
        }

        if (track->algo() == 13 || track->algo() == 14 || track->originalAlgo() == 13 || track->originalAlgo() == 14) {
          if (sig3d > maxSigLoose_) {
            if (verbose_)
              std::cout << " bad muon-seeded track impact parameter, pT = " << pfCandidate.pt()
                        << ", dxySig = " << dxySig << ", dzSig = " << dzSig << std::endl;
            foundBadCandidate = true;
            pBadCandidateCollection->push_back(pfCandidate);
            continue;
          }

          if (nHits < minTrackNHits_) {
            if (verbose_)
              std::cout << " bad muon-seeded track with small nhits, pT = " << pfCandidate.pt() << ", nhits = " << nHits
                        << std::endl;
            foundBadCandidate = true;
            pBadCandidateCollection->push_back(pfCandidate);
            continue;
          }
        }

        double caloEnergy = pfCandidate.ecalEnergy() + pfCandidate.hcalEnergy();

        if (caloEnergy < track->p() * minCaloCompatibility_) {
          if (sig3d > maxSigTight_) {
            if (verbose_)
              std::cout << " bad calo-incompatible track impact parameter, pT = " << pfCandidate.pt()
                        << ", dxySig = " << dxySig << ", dzSig = " << dzSig << std::endl;
            foundBadCandidate = true;
            pBadCandidateCollection->push_back(pfCandidate);
            continue;
          }

          if (nHits < minTrackNHits_) {
            if (verbose_)
              std::cout << " bad calo-incompatible track with small nhits, pT = " << pfCandidate.pt()
                        << ", nhits = " << nHits << std::endl;
            foundBadCandidate = true;
            pBadCandidateCollection->push_back(pfCandidate);
            continue;
          }

          if (nPixelHits < minPixelNHits_) {
            if (verbose_)
              std::cout << " bad calo-incompatible track with small nPixhits, pT = " << pfCandidate.pt()
                        << ", nhits = " << nPixelHits << std::endl;
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
  iEvent.put(std::move(pBadCandidateCollection), "cleaned");

  iEvent.put(std::make_unique<bool>(pass));
}

//define this as a plug-in
DEFINE_FWK_MODULE(HiBadParticleFilter);
