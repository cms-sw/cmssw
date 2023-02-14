#include <list>
#include <vector>
#include <limits>
#include <string>
#include <atomic>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Candidate/interface/VertexCompositeCandidate.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"

#include "RecoTracker/DisplacedRegionalTracking/plugins/DisplacedVertexCluster.h"

using namespace std;
typedef DisplacedVertexCluster::Distance Distance;
typedef DisplacedVertexCluster::DistanceItr DistanceItr;

class DisplacedRegionSeedingVertexProducer : public edm::global::EDProducer<> {
public:
  DisplacedRegionSeedingVertexProducer(const edm::ParameterSet &);
  ~DisplacedRegionSeedingVertexProducer() override;
  void produce(edm::StreamID, edm::Event &, const edm::EventSetup &) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions &);

private:
  // clustering parameters
  const double rParam_;

  // selection parameters
  const double minRadius_;
  const double nearThreshold_;
  const double farThreshold_;
  const double discriminatorCut_;
  const vector<string> input_names_;
  const vector<string> output_names_;

  const edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;
  const edm::EDGetTokenT<edm::View<reco::VertexCompositeCandidate> > trackClustersToken_;

  tensorflow::Session *session_;

  double getDiscriminatorValue(const DisplacedVertexCluster &, const reco::BeamSpot &) const;
};

DisplacedRegionSeedingVertexProducer::DisplacedRegionSeedingVertexProducer(const edm::ParameterSet &cfg)
    : rParam_(cfg.getParameter<double>("rParam")),
      minRadius_(cfg.getParameter<double>("minRadius")),
      nearThreshold_(cfg.getParameter<double>("nearThreshold")),
      farThreshold_(cfg.getParameter<double>("farThreshold")),
      discriminatorCut_(cfg.getParameter<double>("discriminatorCut")),
      input_names_(cfg.getParameter<vector<string> >("input_names")),
      output_names_(cfg.getParameter<vector<string> >("output_names")),
      beamSpotToken_(consumes<reco::BeamSpot>(cfg.getParameter<edm::InputTag>("beamSpot"))),
      trackClustersToken_(
          consumes<edm::View<reco::VertexCompositeCandidate> >(cfg.getParameter<edm::InputTag>("trackClusters"))) {
  unsigned nThreads = cfg.getUntrackedParameter<unsigned>("nThreads");
  tensorflow::SessionOptions sessionOptions;
  tensorflow::setThreading(sessionOptions, nThreads);
  string pbFile = cfg.getParameter<edm::FileInPath>("graph_path").fullPath();
  auto graphDef = tensorflow::loadGraphDef(pbFile);
  session_ = tensorflow::createSession(graphDef, sessionOptions);

  produces<vector<reco::Vertex> >("nearRegionsOfInterest");
  produces<vector<reco::Vertex> >("farRegionsOfInterest");
}

DisplacedRegionSeedingVertexProducer::~DisplacedRegionSeedingVertexProducer() {
  if (session_ != nullptr)
    tensorflow::closeSession(session_);
}

void DisplacedRegionSeedingVertexProducer::produce(edm::StreamID streamID,
                                                   edm::Event &event,
                                                   const edm::EventSetup &setup) const {
  const auto &beamSpot = event.get(beamSpotToken_);
  const math::XYZVector bs(beamSpot.position());

  const auto &trackClusters = event.get(trackClustersToken_);

  // Initialize distances.
  list<DisplacedVertexCluster> pseudoROIs;
  list<Distance> distances;
  const double minTrackClusterRadius = minRadius_ - rParam_;
  for (unsigned i = 0; i < trackClusters.size(); i++) {
    const reco::VertexCompositeCandidate &trackCluster = trackClusters[i];
    const math::XYZVector x(trackCluster.vertex());
    if (minRadius_ < 0.0 || minTrackClusterRadius < 0.0 || (x - bs).rho() > minTrackClusterRadius)
      pseudoROIs.emplace_back(&trackClusters.at(i), rParam_);
  }
  if (pseudoROIs.size() > 1) {
    DisplacedVertexClusterItr secondToLast = pseudoROIs.end();
    secondToLast--;
    for (DisplacedVertexClusterItr i = pseudoROIs.begin(); i != secondToLast; i++) {
      DisplacedVertexClusterItr j = i;
      j++;
      for (; j != pseudoROIs.end(); j++) {
        distances.emplace_back(i, j);

        // Track clusters farther apart than 4 times rParam_ (i.e., 16 times
        // rParam_^2) cannot wind up in the same ROI, so remove these pairs.
        if (distances.back().distance2() > 16.0 * rParam_ * rParam_)
          distances.pop_back();
      }
    }
  }

  // Do clustering.
  while (!distances.empty()) {
    const auto comp = [](const Distance &a, const Distance &b) { return a.distance2() <= b.distance2(); };
    distances.sort(comp);
    DistanceItr dBest = distances.begin();
    if (dBest->distance2() > rParam_ * rParam_)
      break;

    dBest->entities().first->merge(*dBest->entities().second);
    dBest->entities().second->setInvalid();

    const auto distancePred = [](const Distance &a) {
      return (!a.entities().first->valid() || !a.entities().second->valid());
    };
    const auto pseudoROIPred = [](const DisplacedVertexCluster &a) { return !a.valid(); };
    distances.remove_if(distancePred);
    pseudoROIs.remove_if(pseudoROIPred);
  }

  // Remove invalid ROIs.
  const auto roiPred = [&](const DisplacedVertexCluster &roi) {
    if (!roi.valid())
      return true;
    const auto &x(roi.centerOfMass());
    if ((x - bs).rho() < minRadius_)
      return true;
    const double discriminatorValue = ((discriminatorCut_ > 0.0) ? getDiscriminatorValue(roi, beamSpot) : 1.0);
    if (discriminatorValue < discriminatorCut_)
      return true;
    return false;
  };
  pseudoROIs.remove_if(roiPred);

  auto nearRegionsOfInterest = make_unique<vector<reco::Vertex> >();
  auto farRegionsOfInterest = make_unique<vector<reco::Vertex> >();

  constexpr std::array<double, 6> errorA{{1.0, 0.0, 1.0, 0.0, 0.0, 1.0}};
  static const reco::Vertex::Error errorRegion(errorA.begin(), errorA.end(), true, true);

  for (const auto &roi : pseudoROIs) {
    const auto &x(roi.centerOfMass());
    if ((x - bs).rho() < nearThreshold_)
      nearRegionsOfInterest->emplace_back(reco::Vertex::Point(roi.centerOfMass()), errorRegion);
    if ((x - bs).rho() > farThreshold_)
      farRegionsOfInterest->emplace_back(reco::Vertex::Point(roi.centerOfMass()), errorRegion);
  }

  event.put(move(nearRegionsOfInterest), "nearRegionsOfInterest");
  event.put(move(farRegionsOfInterest), "farRegionsOfInterest");
}

void DisplacedRegionSeedingVertexProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<double>("rParam", 1.0);
  desc.add<double>("minRadius", -1.0);
  desc.add<double>("nearThreshold", 9999.0);
  desc.add<double>("farThreshold", -1.0);
  desc.add<double>("discriminatorCut", -1.0);
  desc.add<vector<string> >("input_names", {"phi_0", "phi_1"});
  desc.add<vector<string> >("output_names", {"model_5/activation_10/Softmax"});
  desc.addUntracked<unsigned>("nThreads", 1);
  desc.add<edm::FileInPath>(
      "graph_path",
      edm::FileInPath(
          "RecoTracker/DisplacedRegionalTracking/data/FullData_Phi-64-128-256_16-32-64_F-128-64-32_Model.pb"));
  desc.add<edm::InputTag>("beamSpot", edm::InputTag("offlineBeamSpot"));
  desc.add<edm::InputTag>("trackClusters", edm::InputTag("generalV0Candidates", "Kshort"));

  descriptions.add("displacedRegionProducer", desc);
}

double DisplacedRegionSeedingVertexProducer::getDiscriminatorValue(const DisplacedVertexCluster &roi,
                                                                   const reco::BeamSpot &bs) const {
  // The network takes in two maps of data features, one with information
  // related to the pairwise track vertices and one with information related to
  // the tracks in an isolation annulus.

  constexpr int maxNVertices = 40;     // maximum number of pairwise track vertices per ROI
  constexpr int nVertexFeatures = 23;  // number of features per pairwise track vertex
  constexpr int nDimVertices = 3;      // number of tensor dimensions for the map of pairwise track vertices

  constexpr int maxNAnnulusTracks = 10;     // maximum number of annulus tracks per ROI
  constexpr int nAnnulusTrackFeatures = 8;  // number of features per annulus track
  constexpr int nDimAnnulusTracks = 3;      // number of tensor dimensions for the map of annulus tracks

  tensorflow::Tensor vertexTensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, maxNVertices, nVertexFeatures}));
  auto vertex_map = vertexTensor.tensor<float, nDimVertices>();
  tensorflow::Tensor annulusTensor(tensorflow::DT_FLOAT,
                                   tensorflow::TensorShape({1, maxNAnnulusTracks, nAnnulusTrackFeatures}));
  auto annulus_map = annulusTensor.tensor<float, nDimAnnulusTracks>();

  for (int i = 0, map_i = 0; map_i < maxNVertices; i++, map_i++) {
    if (i >= static_cast<int>(roi.nConstituents()))
      for (unsigned j = 0; j < nVertexFeatures; j++)
        vertex_map(0, map_i, j) = 0.0;
    else {
      const auto &trackCluster = *roi.constituent(i);
      const auto &track0 = *trackCluster.daughter(0)->bestTrack();
      const auto &track1 = *trackCluster.daughter(1)->bestTrack();

      vertex_map(0, map_i, 0) = trackCluster.vx() - bs.x0();
      vertex_map(0, map_i, 1) = trackCluster.vy() - bs.y0();
      vertex_map(0, map_i, 2) = trackCluster.vz() - bs.z0();

      vertex_map(0, map_i, 3) = trackCluster.vertexCovariance()(0, 0);
      vertex_map(0, map_i, 4) = trackCluster.vertexCovariance()(0, 1);
      vertex_map(0, map_i, 5) = trackCluster.vertexCovariance()(0, 2);
      vertex_map(0, map_i, 6) = trackCluster.vertexCovariance()(1, 1);
      vertex_map(0, map_i, 7) = trackCluster.vertexCovariance()(1, 2);
      vertex_map(0, map_i, 8) = trackCluster.vertexCovariance()(2, 2);

      vertex_map(0, map_i, 9) = track0.charge() * track0.pt();
      vertex_map(0, map_i, 10) = track0.eta();
      vertex_map(0, map_i, 11) = track0.phi();
      vertex_map(0, map_i, 12) = track0.dxy(bs);
      vertex_map(0, map_i, 13) = track0.dz(bs.position());
      vertex_map(0, map_i, 14) = track0.normalizedChi2();
      vertex_map(0, map_i, 15) = track0.quality(reco::Track::highPurity) ? 1 : 0;

      vertex_map(0, map_i, 16) = track1.charge() * track1.pt();
      vertex_map(0, map_i, 17) = track1.eta();
      vertex_map(0, map_i, 18) = track1.phi();
      vertex_map(0, map_i, 19) = track1.dxy(bs);
      vertex_map(0, map_i, 20) = track1.dz(bs.position());
      vertex_map(0, map_i, 21) = track1.normalizedChi2();
      vertex_map(0, map_i, 22) = track1.quality(reco::Track::highPurity) ? 1 : 0;
    }
  }

  for (int i = 0; i < maxNAnnulusTracks; i++)
    for (unsigned j = 0; j < nAnnulusTrackFeatures; j++)
      annulus_map(0, i, j) = 0.0;

  tensorflow::NamedTensorList input_tensors;
  input_tensors.resize(2);
  input_tensors[0] = tensorflow::NamedTensor(input_names_.at(0), vertexTensor);
  input_tensors[1] = tensorflow::NamedTensor(input_names_.at(1), annulusTensor);
  vector<tensorflow::Tensor> outputs;
  tensorflow::run(session_, input_tensors, output_names_, &outputs);

  return (outputs.at(0).flat<float>()(1));
}

DEFINE_FWK_MODULE(DisplacedRegionSeedingVertexProducer);
