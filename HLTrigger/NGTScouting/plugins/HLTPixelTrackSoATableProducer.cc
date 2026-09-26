#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "DataFormats/NanoAOD/interface/FlatTable.h"
#include "DataFormats/TrackSoA/interface/TracksHost.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/StreamID.h"

namespace {
  // Indices to the 5-dimensional track state vector (CMS convention: {phi, tip, q/pT, cotTheta, zip})
  constexpr auto kStatePhi = 0;
  constexpr auto kStateDxy = 1;
  constexpr auto kStateQOverPt = 2;  // signed q/pT (carries the charge sign)
  constexpr auto kStateDz = 4;

  // Indices into the 5x5 track covariance matrix (CMS convention)
  constexpr auto kCovPhiPhi = 0;             // (0,0)
  constexpr auto kCovPhiDxy = 1;             // (0,1)
  constexpr auto kCovPhiQOverPt = 2;         // (0,2)
  constexpr auto kCovDxyDxy = 5;             // (1,1)
  constexpr auto kCovDxyQOverPt = 6;         // (1,2)
  constexpr auto kCovQOverPtQOverPt = 9;     // (2,2)
  constexpr auto kCovCotThetaCotTheta = 12;  // (3,3)
  constexpr auto kCovCotThetaDz = 13;        // (3,4)
  constexpr auto kCovDzDz = 14;              // (4,4)
}  // namespace

class HLTPixelTrackSoATableProducer : public edm::global::EDProducer<> {
  using TrackSoAHost = reco::TracksHost;

public:
  explicit HLTPixelTrackSoATableProducer(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  const bool skipNonExistingSrc_;
  const std::string name_;
  const std::string doc_;
  const edm::EDGetTokenT<TrackSoAHost> trackSoAToken_;
};

HLTPixelTrackSoATableProducer::HLTPixelTrackSoATableProducer(const edm::ParameterSet& iConfig)
    : skipNonExistingSrc_(iConfig.getParameter<bool>("skipNonExistingSrc")),
      name_(iConfig.getParameter<std::string>("name")),
      doc_(iConfig.getParameter<std::string>("doc")),
      trackSoAToken_(consumes<TrackSoAHost>(iConfig.getParameter<edm::InputTag>("src"))) {
  produces<nanoaod::FlatTable>();
}

void HLTPixelTrackSoATableProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup&) const {
  auto tracksIn = iEvent.getHandle(trackSoAToken_);
  const bool valid = tracksIn.isValid();
  if (!valid && !skipNonExistingSrc_)
    edm::LogWarning("HLTPixelTrackSoATableProducer") << " Invalid handle for " << name_ << " in track SoA input";

  std::vector<float> v_chi2, v_dzError, v_dxyError, v_eta, v_nHits, v_phi, v_phiError, v_pt, v_qOverPtError;
  std::vector<float> v_qOverPt, v_charge, v_dzBS, v_dxyBS, v_nLayers, v_cotThetaError, v_covCotThetaDz;
  std::vector<float> v_covDxyQOverPt, v_covPhiDxy, v_covPhiQOverPt;

  unsigned int nTracks = 0;
  if (valid) {
    const auto view = tracksIn->const_view();
    const reco::TrackSoAConstView& tracks = view.tracks();
    nTracks = tracks.nTracks();

    for (auto* v : {&v_chi2,
                    &v_dzError,
                    &v_dxyError,
                    &v_eta,
                    &v_nHits,
                    &v_phi,
                    &v_phiError,
                    &v_pt,
                    &v_qOverPtError,
                    &v_qOverPt,
                    &v_charge,
                    &v_dzBS,
                    &v_dxyBS,
                    &v_nLayers,
                    &v_cotThetaError,
                    &v_covCotThetaDz,
                    &v_covDxyQOverPt,
                    &v_covPhiDxy,
                    &v_covPhiQOverPt})
      v->resize(nTracks);

    for (unsigned int i = 0; i < nTracks; ++i) {
      const auto& track = tracks[i];
      const auto& cov = track.covariance();
      const auto& state = track.state();

      v_chi2[i] = track.chi2();  // in the SoA chi2 is stored as chi2/ndof
      v_dzError[i] = std::sqrt(cov(kCovDzDz));
      v_dxyError[i] = std::sqrt(cov(kCovDxyDxy));
      v_eta[i] = track.eta();
      v_nHits[i] = reco::nHits(tracks, i);
      v_phi[i] = state(kStatePhi);
      v_phiError[i] = std::sqrt(cov(kCovPhiPhi));
      v_pt[i] = track.pt();
      v_qOverPtError[i] = std::sqrt(cov(kCovQOverPtQOverPt));
      v_qOverPt[i] = state(kStateQOverPt);    // signed q/pT (state(2))
      v_charge[i] = reco::charge(tracks, i);  // charge = sign(state(2))
      v_dzBS[i] = state(kStateDz);
      v_dxyBS[i] = state(kStateDxy);
      v_nLayers[i] = track.nLayers();
      v_cotThetaError[i] = std::sqrt(cov(kCovCotThetaCotTheta));
      v_covCotThetaDz[i] = cov(kCovCotThetaDz);
      v_covDxyQOverPt[i] = cov(kCovDxyQOverPt);
      v_covPhiDxy[i] = cov(kCovPhiDxy);
      v_covPhiQOverPt[i] = cov(kCovPhiQOverPt);
    }
  }

  auto out = std::make_unique<nanoaod::FlatTable>(nTracks, name_, /*singleton*/ false, /*extension*/ false);
  out->setDoc(doc_);
  // The docs are the expressions the equivalent StringObjectFunction columns used to carry.
  out->addColumn<float>("chi2", v_chi2, "chi2()");
  out->addColumn<float>("dzError", v_dzError, "dzError()");
  out->addColumn<float>("dxyError", v_dxyError, "dxyError()");
  out->addColumn<float>("eta", v_eta, "eta()");
  out->addColumn<float>("nHits", v_nHits, "nHits()");
  out->addColumn<float>("phi", v_phi, "phi()");
  out->addColumn<float>("phiError", v_phiError, "phiError()");
  out->addColumn<float>("pt", v_pt, "pt()");
  out->addColumn<float>("qOverPtError", v_qOverPtError, "qOverPtError()");
  out->addColumn<float>("qOverPt", v_qOverPt, "qOverPt()");
  out->addColumn<float>("charge", v_charge, "charge()");
  out->addColumn<float>("dzBS", v_dzBS, "dzBS()");
  out->addColumn<float>("dxyBS", v_dxyBS, "dxyBS()");
  out->addColumn<float>("nLayers", v_nLayers, "nLayers()");
  out->addColumn<float>("cotThetaError", v_cotThetaError, "cotThetaError()");
  out->addColumn<float>("covCotThetaDz", v_covCotThetaDz, "covCotThetaDz()");
  out->addColumn<float>("covDxyQOverPt", v_covDxyQOverPt, "covDxyQOverPt()");
  out->addColumn<float>("covPhiDxy", v_covPhiDxy, "covPhiDxy()");
  out->addColumn<float>("covPhiQOverPt", v_covPhiQOverPt, "covPhiQOverPt()");

  iEvent.put(std::move(out));
}

void HLTPixelTrackSoATableProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<bool>("skipNonExistingSrc", false)
      ->setComment("whether or not to skip producing the table on absent input product");
  desc.add<edm::InputTag>("src", edm::InputTag("pixelTracksAlpaka"))->setComment("reco::TracksHost input collection");
  desc.add<std::string>("name", "PixelTrackSoA")->setComment("name of the flat table output");
  desc.add<std::string>("doc", "Pixel tracks from TrackSoA")->setComment("documentation of the flat table");

  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(HLTPixelTrackSoATableProducer);
