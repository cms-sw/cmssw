#include <cmath>
#include <string>
#include <queue>
#include <cassert>
#include <cmath>
#include <random>
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
                                         const edm::Handle<TICLGraph<ElementaryNode>> &tgH,
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
  else if (n == k)
    return 1;
  else
    return factorial(n) / (factorial(k) * factorial(n - k));
}

//quality function, Constant Potts Model
template <class T>
double CPM(Partition<T> const &partition, double gamma) {
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
double CPM_after_move(Partition<T> const &partition,
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
auto queueCommunity(std::vector<Node<T>> &community, std::queue<Node<T>> const &queue) {
  std::random_shuffle(community.begin(), community.end());  //elements are added to the queue in random order
  for (auto const &node : community) {
    queue.push(node);
  }
  return queue;
}

template <class T>
auto moveNodesFast(TICLGraph<T> const &graph, Partition<T> &partition, double gamma) {
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
Partition<T> &singletonPartition(TICLGraph<T> const &graph, Partition<T> &singlePartition) {
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
  assert(edges >= 0);
  int nodeSize{communitySize(singletonCommunity)};
  int subsetSize{communitySize(subset)};
  return (edges >= (gamma * nodeSize * (subsetSize - nodeSize)));
}

template <class T>
bool isCommunityWellConnected(std::vector<Node<T>> &community, std::vector<Node<T>> &subset, double gamma) {
  std::vector<Node<T>> subsetMinuscommunity{};
  for (auto const &node : subset) {
    auto it{std::find(community.begin(), community.end(), node)};
    if (it == community.end()) {
      subsetMinuscommunity.push_back(node);
    }
  }
  int edges{numberOfEdges(community, subsetMinuscommunity)};
  assert(edges >= 0);
  int comSize{communitySize(community)};
  int subsetSize{communitySize(subset)};
  return (edges >= (gamma * comSize * (subsetSize - comSize)));
}

template <class T>
int extractRandomCommunityIndex(std::vector<std::vector<Node<T>>> const &communities,
                                Partition<T> const &partition,
                                Node<T> const &node,
                                std::vector<Node<T>> nodeCommunity,
                                double theta) {
  auto currentCPM{CPM(partition, gamma)};
  std::vector<double> deltaCPMs{};

  //calculating delta_H for all communities
  for (auto const &community : communities) {
    if (isCommunityWellConnected(community)) {
      double afterMoveCPM{CPM_after_move(partition, gamma, nodeCommunity, community, node)};
      deltaCPMs.push_back((afterMoveCPM - currentCPM));
    }
  }

  //creating the discrete probability function
  std::vector<double> distribution{};
  for (auto const &deltaCPM : deltaCPMs) {
    if (deltaCPM < 0) {
      distribution.push_back(0.);
    } else {
      assert(theta > 0);
      distribution.push_back(std::exp(deltaCPM / theta));
    }
  }

  //extracting a random community
  std::random_device rd;
  std::mt19937 gen(rd());
  std::discrete_distribution<> d(distribution.begin(), distribution.end());
  //extracts a random index
  int resultIndex = d(gen);

  return resultIndex;
}

template <class T>
Partition<T> &mergeNodesSubset(
    TICLGraph<T> const &graph, Partition<T> &partition, std::vector<Node<T>> &subset, double gamma, double theta) {
  auto &communities{partition.setPartition()};

  for (auto const &node : subset) {
    if (isNodeWellConnected(node, subset, gamma)) {
      int index{findCommunityIndex(node)};
      auto &nodeCommunity{communities[index]};

      assert((communitySize(nodeCommunity)) != 0);
      if (communitySize(nodeCommunity) == 1) {
        int communityToIndex{extractRandomCommunityIndex(communities, partition, node, nodeCommunity, theta)};
        auto &communityTo{communities[communityToIndex]};
        moveNode(nodeCommunity, communityTo, node);
      }
    }
  }

  return partition;
}

//necessary to do before calling refinePartition bc if I just use an empty vector no parameter template deduction is possible
//std::vector<Node<T>> singleCommunities{partition.getPartition()};
// singleCommunities.clear();
// Partition<T> singlePartition{singleCommunities};
template <class T>
Partition<T> &refinePartition(TICLGraph<T> const &graph, Partition<T> &partition, Partition<T> &singlePartition) {
  //fills an empty partition with a singleton partition
  auto &refinedPartition{singletonPartition(graph, singlePartition)};
  auto const &communities{partition.getPartition()};
  for (auto const &community : communities) {
    mergeNodesSubset(graph, refinedPartition, community);
  }
  return refinedPartition;
}

//***********PROBLEM HERE: I DONT LIKE RETURNING IT AS COPY but im not sure it can be done otherwise*****************
template <class T>
TICLGraph<Node<std::vector<Node<T>>>> aggregateGraph(Partition<T> const &partition) {
  //communities become nodes in aggregate graph
  std::vector<std::vector<Node<T>>> const &communities{partition.getPartition()};
  Node<std::vector<Node<T>>> firstAggregateNode{communities[0]};
  std::vector<Node<std::vector<Node<T>>>> aggregateNodes{firstAggregateNode};

  std::for_each((communities.begin() + 1), communities.end(), [&aggregateNodes](std::vector<Node<T>> const &community) {
    Node<std::vector<Node<T>>> aggregateNode{community};
    aggregateNodes.push_back(aggregateNode);
  });

  assert(aggregateNodes.size() == communities.size());
  TICLGraph<Node<std::vector<Node<T>>>> aggregateGraph{aggregateNodes};
  return aggregateGraph;
}
