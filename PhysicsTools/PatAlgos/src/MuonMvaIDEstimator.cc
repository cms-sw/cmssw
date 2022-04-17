#include "PhysicsTools/PatAlgos/interface/MuonMvaIDEstimator.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/PatCandidates/interface/Muon.h"

using namespace pat;
using namespace cms::Ort;

MuonMvaIDEstimator::MuonMvaIDEstimator(const edm::FileInPath &weightsfile) {
  randomForest_ = std::make_unique<ONNXRuntime>(weightsfile.fullPath());
  LogDebug("MuonMvaIDEstimator") << randomForest_.get();
}

void MuonMvaIDEstimator::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::FileInPath>("mvaIDTrainingFile", edm::FileInPath("RecoMuon/MuonIdentification/data/mvaID.onnx"));
  desc.add<std::vector<std::string>>("flav_names",
                                     std::vector<std::string>{
                                         "probBAD",
                                         "probGOOD",
                                     });

  descriptions.addWithDefaultLabel(desc);
}

void MuonMvaIDEstimator::globalEndJob(const cms::Ort::ONNXRuntime *cache) {}
const reco::Muon::ArbitrationType arbitrationType = reco::Muon::SegmentAndTrackArbitration;
std::vector<float> MuonMvaIDEstimator::computeMVAID(const pat::Muon &muon) const {
  const float local_chi2 = muon.combinedQuality().chi2LocalPosition;
  const float kink = muon.combinedQuality().trkKink;
  const float segment_comp = muon.segmentCompatibility(arbitrationType);
  const float n_MatchedStations = muon.numberOfMatchedStations();
  const float pt = muon.pt();
  const float eta = muon.eta();
  const float global_muon = muon.isGlobalMuon();
  float Valid_pixel;
  float tracker_layers;
  float validFraction;
  if (muon.innerTrack().isNonnull()) {
    Valid_pixel = muon.innerTrack()->hitPattern().numberOfValidPixelHits();
    tracker_layers = muon.innerTrack()->hitPattern().trackerLayersWithMeasurement();
    validFraction = muon.innerTrack()->validFraction();
  } else {
    Valid_pixel = -99.;
    tracker_layers = -99.0;
    validFraction = -99.0;
  }
  float norm_chi2;
  float n_Valid_hits;
  if (muon.globalTrack().isNonnull()) {
    norm_chi2 = muon.globalTrack()->normalizedChi2();
    n_Valid_hits = muon.globalTrack()->hitPattern().numberOfValidMuonHits();
  } else if (muon.innerTrack().isNonnull()) {
    norm_chi2 = muon.innerTrack()->normalizedChi2();
    n_Valid_hits = muon.innerTrack()->hitPattern().numberOfValidMuonHits();
  } else {
    norm_chi2 = -99;
    n_Valid_hits = -99;
  }
  const std::vector<std::string> input_names_{"float_input"};
  std::vector<float> vars = {global_muon,
                             validFraction,
                             norm_chi2,
                             local_chi2,
                             kink,
                             segment_comp,
                             n_Valid_hits,
                             n_MatchedStations,
                             Valid_pixel,
                             tracker_layers,
                             pt,
                             eta};
  const std::vector<std::string> flav_names_{"probBAD", "probGOOD"};
  cms::Ort::FloatArrays input_values_;
  input_values_.emplace_back(vars);
  std::vector<float> outputs;
  LogDebug("MuonMvaIDEstimator") << randomForest_.get();
  outputs = randomForest_->run(input_names_, input_values_, {}, {"probabilities"})[0];
  assert(outputs.size() == flav_names_.size());
  return outputs;
}
