#include <cmath>
#include <string>
#include <queue>
#include <cassert>
#include <cmath>
#include "RecoHGCal/TICL/plugins/LinkingAlgoByLeiden.h"

#include "DataFormats/GeometrySurface/interface/BoundDisk.h"
#include "DataFormats/HGCalReco/interface/Common.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

#include "RecoParticleFlow/PFProducer/interface/PFMuonAlgo.h"

using namespace ticl;

LinkingAlgoByLeiden::LinkingAlgoByLeiden(const edm::ParameterSet &conf)
    : LinkingAlgoBase(conf), cutTk_(conf.getParameter<std::string>("cutTk")) {}

LinkingAlgoByLeiden::~LinkingAlgoByLeiden() {}

void LinkingAlgoByLeiden::buildLayers() {
  // build disks at HGCal front & EM-Had interface for track propagation

  float zVal = hgcons_->waferZ(1, true);
  std::pair<float, float> rMinMax = hgcons_->rangeR(zVal, true);

  float zVal_interface = rhtools_.getPositionLayer(rhtools_.lastLayerEE()).z();
  std::pair<float, float> rMinMax_interface = hgcons_->rangeR(zVal_interface, true);

  for (int iSide = 0; iSide < 2; ++iSide) {
    float zSide = (iSide == 0) ? (-1. * zVal) : zVal;
    firstDisk_[iSide] =
        std::make_unique<GeomDet>(Disk::build(Disk::PositionType(0, 0, zSide),
                                              Disk::RotationType(),
                                              SimpleDiskBounds(rMinMax.first, rMinMax.second, zSide - 0.5, zSide + 0.5))
                                      .get());

    zSide = (iSide == 0) ? (-1. * zVal_interface) : zVal_interface;
    interfaceDisk_[iSide] = std::make_unique<GeomDet>(
        Disk::build(Disk::PositionType(0, 0, zSide),
                    Disk::RotationType(),
                    SimpleDiskBounds(rMinMax_interface.first, rMinMax_interface.second, zSide - 0.5, zSide + 0.5))
            .get());
  }
}

void LinkingAlgoByLeiden::initialize(const HGCalDDDConstants *hgcons,
                                     const hgcal::RecHitTools rhtools,
                                     const edm::ESHandle<MagneticField> bfieldH,
                                     const edm::ESHandle<Propagator> propH) {
  hgcons_ = hgcons;
  rhtools_ = rhtools;
  buildLayers();

  bfield_ = bfieldH;
  propagator_ = propH;
}

void LinkingAlgoByLeiden::linkTracksters(const edm::Handle<std::vector<reco::Track>> tkH,
                                         const edm::Handle<edm::ValueMap<float>> tkTime_h,
                                         const edm::Handle<edm::ValueMap<float>> tkTimeErr_h,
                                         const edm::Handle<edm::ValueMap<float>> tkTimeQual_h,
                                         const std::vector<reco::Muon> &muons,
                                         const edm::Handle<std::vector<Trackster>> tsH,
                                         const edm::Handle<TICLGraph> &tgH,
                                         const bool useMTDTiming,
                                         std::vector<TICLCandidate> &resultLinked,
                                         std::vector<TICLCandidate> &chargedHadronsFromTk) {
  std::cout << "Il mio bellissimo algoritmo" << '\n';
}

void LinkingAlgoByLeiden::fillPSetDescription(edm::ParameterSetDescription &desc) {
  desc.add<std::string>("cutTk",
                        "1.48 < abs(eta) < 3.0 && pt > 1. && quality(\"highPurity\") && "
                        "hitPattern().numberOfLostHits(\"MISSING_OUTER_HITS\") < 5");
  LinkingAlgoBase::fillPSetDescription(desc);
}

template <class T>
auto moveNodesFast(TICLGraph const &graph, Partition &partition) {
  auto communities{partition.setPartition()};
  std::random_shuffle(communities.begin(), communities.end());
  std::queue<Node<T>> queue{};
  std::vector<Node<T>> empty_community{};

  for (auto &community : communities) {  //all nodes are added to queue in random order
    queueCommunity(community, queue);
  }

  while (!queue.empty()) {
    auto current_node{queue.front()};
  }

  //**********NEEDS IMPLEMENTATION**********

  return partition;
}

template <class T>
auto queueCommunity(std::vector<Node<T>> &community, std::queue const &queue) {
  std::random_shuffle(community.begin(), community.end());  //elements are added to the queue in random order
  for (auto const &node : community) {
    queue.push(node);
  }
  return queue;
}

//quality function, Constant Potts Model
template <class T>
auto CPM(Partition &partition, double const gamma) {
  double CPMResult{};
  for (auto const &community : partition) {
    CPMResult += (numberOfEdges(community, community) - gamma * binomialCoefficient(communitySize(community), 2));
  }
  return CPMResult;
}

int factorial(int n) { return (n == 1 || n == 0) ? 1 : n * factorial(n - 1); }

int binomialCoefficient(int n, int k) {
  assert(n >= 0);
  assert(k >= 0);
  if (n < k)
    return 0;
  else if (n = k)
    return 1;
  else
    return facorial(n) / (factorial(k) * factorial(n - k));
}
