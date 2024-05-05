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

//quality function, Constant Potts Model
template <class T>
double CPM(Partition const &partition, double gamma) {
  double CPMResult{};
  for (auto const &community : partition) {
    CPMResult += (numberOfEdges(community, community) - gamma * binomialCoefficient(communitySize(community), 2));
  }
  return CPMResult;
}

template <class T>
double CPM_contribution_from_new_community(Node<T> const &node, double gamma) {
  std::vector<Node<T>> newCommunity{node};
  double result{(-gamma * binomialCoefficient(communitySize(newCommunity), 2))};
  assert(result <= 0.);
  return result;
}

template <class T>
double CPM_after_move(Partition const &partition,
                      double gamma,
                      std::vector<Node<T>> const &communityFrom,
                      std::vector<Node<T>> const &communityTo,
                      Node<T> const &node) {
  double CPMResult{};
  auto const &communities{partition.getPartition()};
  for (auto const &community : communities) {
    if (community == communityFrom) {
      std::vector<Node<T>> communityWithoutNode{community};
      std::remove(communityWithoutNode.begin(), communityWithoutNode.end(), node);
      communityWithoutNode.pop_back();
      CPMResult += (numberOfEdges(communityWithoutNode, communityWithoutNode) -
                    gamma * binomialCoefficient(communitySize(communityWithoutNode), 2));
    } else if (community == communityTo) {
      std::vector<Node<T>> communityWithNewNode{community};
      communityWithNewNode.push_back(node);
      CPMResult += (numberOfEdges(communityWithNewNode, communityWithNewNode) -
                    gamma * binomialCoefficient(communitySize(communityWithNewNode), 2));
    } else {
      CPMResult += (numberOfEdges(community, community) - gamma * binomialCoefficient(communitySize(community), 2));
    }
  }
  return CPMResult;
}

template <class T>
void moveNode(std::vector<Node<T>> &communityFrom, std::vector<Node<T>> &communityTo, Node<T> const &node) {
  std::remove(communityFrom.begin(), communityFrom.end(), node);
  communityFrom.pop_back();
  communityTo.push_back(node);
}

template <class T>
auto queueCommunity(std::vector<Node<T>> &community, std::queue const &queue) {
  std::random_shuffle(community.begin(), community.end());  //elements are added to the queue in random order
  for (auto const &node : community) {
    queue.push(node);
  }
  return queue;
}

template <class T>
auto moveNodesFast(TICLGraph const &graph, Partition &partition, double gamma) {
  auto shuffledCommunities{partition.getPartition()};
  std::random_shuffle(shuffledCommunities.begin(), shuffledCommunities.end());
  std::queue<Node<T>> queue{};
  //std::vector<Node<T>> empty_community{};

  for (auto &community : shuffledCommunities) {  //all nodes are added to queue in random order
    queueCommunity(community, queue);
  }

  while (!queue.empty()) {
    Node<T> const &currentNode{queue.front()};
    auto currentCPM{CPM(partition, gamma) + CPM_contribution_from_new_community(currentNode, gamma)};
    auto &currentCommunity{partition.findCommunity(currentNode)};
    auto &communities{partition.getPartition()};

    int indexBestCommunity{};
    int iterationIndex{-1};
    double bestDeltaCPM{0.};
    for (auto const &community : communities) {
      ++iterationIndex;
      double AfterMoveCPM{CPM_after_move(partition, gamma, currentCommunity, community, currentNode)};
      double deltaCPM{AfterMoveCPM - currentCPM};
      if (deltaCPM > bestDeltaCPM) {
        bestDeltaCPM = deltaCPM;
        indexBestCommunity = iterationIndex;
      }
    }
    if (bestDeltaCPM > 0.) {
      moveNode(currentCommunity, communities[indexBestCommunity], currentNode);
      std::vector<Node<T>> currentNeighbours{};
      for (auto const &community : communities) {
        if (!(community == communities[indexBestCommunity])) {
          for (auto const &node : community) {
            if (areNeighbours(currentNode, node)) {
              currentNeighbours.push_back(node);
            }
          }
        }
      }
      // making sure all nbrs of currentNode who are not in bestCommunity will be visited
      for (auto const &neighbour : currentNeighbours) {
        queue.push(neighbour);
      }
    }
  }

  return partition;
}

//fills an empty partition with a singleton partition
template <class T>
Partition &singletonPartition(TICLGraph const &graph, Partition &singlePartition) {
  assert((singlePartition.getPartition()).empty());
  auto const &nodes{graph.getNodes()};
  auto &communities{singlePartition.setPartition()};
  for (auto const &node : nodes) {
    std::vector<Node<T>> singletonCommunity{node};
    communities.push_back(singletonCommunity);
  }
  assert(!((singlePartition.getPartition()).empty()));

  return singlePartition;
}

template <class T>
bool isNodeWellConnected(Node<T> const &node, std::vector<Node<T>> &subset, double gamma) {
  std::vector<Node<T>> const singletonCommunity{node};
  int edges{numberOfEdges(singletonCommunity, subset)};
  assert(numberOfEdges >= 0);
  int nodeSize{communitySize(singletonCommunity)};
  int subsetSize{communitySize(subset)};
  return (edges >= (gamma * nodeSize * (subsetSize - nodeSize)));
}

template <class T>
Partition &mergeNodesSubset(TICLGraph const &graph, Partition &partition, std::vector<Node<T>> &subset, double gamma) {
  for (auto const &node : subset) {
    if (isNodeWellConnected(node, subset, gamma)) {
      auto const &nodeCommunity{findCommunity(node)};
      assert((communitySize(nodeCommunity)) != 0);
      if (communitySize(nodeCommunity) == 1) {
        //*************NEEDS IMPLEMENTATION***************
        //needs auxiliary function in TICLGraph.h isContained
        //needs auxiliary function above this one isCommunityWellConnected
      }
    }
  }
}