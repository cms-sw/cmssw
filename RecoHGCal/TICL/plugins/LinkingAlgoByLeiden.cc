#include <cmath>
#include <queue>
#include <cassert>
#include <cmath>
#include <random>
#include <string>

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
  std::cout << "Il mio bellissimo algoritmo";
}

void LinkingAlgoByLeiden::fillPSetDescription(edm::ParameterSetDescription &desc) {
  desc.add<std::string>("cutTk",
                        "1.48 < abs(eta) < 3.0 && pt > 1. && quality(\"highPurity\") && "
                        "hitPattern().numberOfLostHits(\"MISSING_OUTER_HITS\") < 5");
  LinkingAlgoBase::fillPSetDescription(desc);
}

void LinkingAlgoByLeiden::leidenAlgorithm(TICLGraph &graph,
                                          Partition &partition,
                                          std::vector<Flat> &flatFinalPartition) {
  moveNodesFast(partition, gamma_);

  if (!(isAlgorithmDone(graph, partition))) {
    Partition refinedPartition = Partition{std::vector<Community>{}};
    assert((refinedPartition.getCommunities()).empty());

    refinePartition(graph, partition, refinedPartition, gamma_, theta_);
    aggregateGraph(graph, refinedPartition);
    auto &communities = partition.getCommunities();
    std::vector<Community> aggregatedCommunities{};

    for (auto const &community : communities) {
      Community aggregatedCommunity{};
      for (auto const &aggregateNode : graph.getNodes()) {
        if (isCommunityContained(std::get<Community>(aggregateNode), community)) {
          aggregatedCommunity.getNodes().push_back(aggregateNode);
        }
      }
      aggregatedCommunities.push_back(aggregatedCommunity);
    }

    communities = aggregatedCommunities;
    leidenAlgorithm(graph, partition, flatFinalPartition);
  }

  else {
    partition.flattenPartition(flatFinalPartition);
  }
}
bool isAlgorithmDone(TICLGraph const &graph, Partition const &partition) {
  return (partition.getCommunities()).size() == (graph.getNodes()).size();
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
double CPM(Partition const &partition, double gamma) {
  double CPMResult{};
  for (auto const &community : partition.getCommunities()) {
    CPMResult += (numberOfEdges(community, community) - gamma * binomialCoefficient(communitySize(community), 2));
  }
  return CPMResult;
}

double CPM_contribution_from_new_community(Node const &node, double gamma) {
  Community newCommunity{std::vector<Node>{node}, 1};
  double result{(-gamma * binomialCoefficient(communitySize(newCommunity), 2))};
  assert(result <= 0.);
  return result;
}

double CPM_after_move(Partition const &partition,
                      double gamma,
                      Community const &communityFrom,
                      Community const &communityTo,
                      Node const &node) {
  double CPMResult{};
  auto const &communities = partition.getCommunities();
  for (auto const &community : communities) {
    if (community == communityFrom) {
      std::vector<Node> vectorWithoutNode{};
      std::copy_if(communityFrom.getNodes().begin(),
                   communityFrom.getNodes().end(),
                   std::back_inserter(vectorWithoutNode),
                   [&](Node const &n) { return !(n == node); });
      Community communityWithoutNode{vectorWithoutNode, communityFrom.getDegree()};
      CPMResult += (numberOfEdges(communityWithoutNode, communityWithoutNode) -
                    gamma * binomialCoefficient(communitySize(communityWithoutNode), 2));
    } else if (community == communityTo) {
      Community communityWithNewNode{community};
      communityWithNewNode.getNodes().push_back(node);
      CPMResult += (numberOfEdges(communityWithNewNode, communityWithNewNode) -
                    gamma * binomialCoefficient(communitySize(communityWithNewNode), 2));
    } else {
      CPMResult += (numberOfEdges(community, community) - gamma * binomialCoefficient(communitySize(community), 2));
    }
  }
  return CPMResult;
}

void moveNode(Community &communityFrom, Community &communityTo, Node const &node) {
  communityFrom.getNodes().erase(std::remove(communityFrom.getNodes().begin(), communityFrom.getNodes().end(), node));
  communityTo.getNodes().push_back(node);
}

auto queueCommunity(Community &community, std::queue<Node> &queue) {
  //elements are added to the queue in random order
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(community.getNodes().begin(), community.getNodes().end(), g);

  for (auto const &node : community.getNodes()) {
    queue.push(node);
  }
  return queue;
}

Partition &removeEmptyCommunities(Partition &partition) {
  auto &communities = partition.getCommunities();
  communities.erase(std::remove_if(communities.begin(), communities.end(), [](Community const &community) {
    return community.getNodes().size() == 0;
  }));

  auto const &communitiesAfterRemoval = partition.getCommunities();
  for (auto const &communityAfterRemoval : communitiesAfterRemoval) {
    assert(communityAfterRemoval.getNodes().size() != 0);
  }

  return partition;
}

Partition &moveNodesFast(Partition &partition, double gamma) {
  auto shuffledCommunities = partition.getCommunities();
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(shuffledCommunities.begin(), shuffledCommunities.end(), g);
  std::queue<Node> queue{};
  //std::vector<Node<T>> empty_community{};

  for (auto &community : shuffledCommunities) {  //all nodes are added to queue in random order
    queueCommunity(community, queue);
  }

  while (!queue.empty()) {
    Node const &currentNode{queue.front()};
    auto currentCPM = CPM(partition, gamma) + CPM_contribution_from_new_community(currentNode, gamma);
    auto &currentCommunity = partition.getCommunities()[partition.findCommunityIndex(currentNode)];
    auto &communities = partition.getCommunities();

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
      std::vector<Node> currentNeighbours{};
      for (auto const &community : communities) {
        if (!(community == communities[indexBestCommunity])) {
          for (auto const &node : community.getNodes()) {
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

  //remove communities that, after node moving, are empty and then returns the result
  return removeEmptyCommunities(partition);
}

//fills an empty partition with a singleton partition
Partition &singletonPartition(TICLGraph const &graph, Partition &singlePartition) {
  assert(singlePartition.getCommunities().empty());
  auto const &nodes = graph.getNodes();
  auto &communities = singlePartition.getCommunities();
  for (auto const &node : nodes) {
    Community singletonCommunity{std::vector{node}, degree(node) + 1};
    communities.push_back(singletonCommunity);
  }
  assert(!(singlePartition.getCommunities().empty()));

  return singlePartition;
}

bool isNodeWellConnected(Node const &node, Community const &subset, double gamma) {
  Community singletonCommunity{std::vector{node}, degree(node) + 1};
  int edges{numberOfEdges(singletonCommunity, subset)};
  assert(edges >= 0);
  int nodeSize{communitySize(singletonCommunity)};
  int subsetSize{communitySize(subset)};
  return (edges >= (gamma * nodeSize * (subsetSize - nodeSize)));
}

bool isCommunityWellConnected(Community const &community, Community const &subset, double gamma) {
  Community subsetMinuscommunity{};
  for (auto const &node : subset.getNodes()) {
    auto it = std::find(community.getNodes().begin(), community.getNodes().end(), node);
    if (it == community.getNodes().end()) {
      subsetMinuscommunity.getNodes().push_back(node);
    }
  }
  int edges{numberOfEdges(community, subsetMinuscommunity)};
  assert(edges >= 0);
  int comSize{communitySize(community)};
  int subsetSize{communitySize(subset)};
  return (edges >= (gamma * comSize * (subsetSize - comSize)));
}

int extractRandomCommunityIndex(std::vector<Community> const &communities,
                                Partition const &partition,
                                Node const &node,
                                Community const &nodeCommunity,
                                Community const &subset,
                                double gamma,
                                double theta) {
  double currentCPM = CPM(partition, gamma);
  std::vector<double> deltaCPMs{};

  //calculating delta_H for all communities
  for (auto const &community : communities) {
    if (isCommunityWellConnected(community, subset, gamma)) {
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

//arrived here atm

Partition &mergeNodesSubset(Partition &partition, Community const &subset, double gamma, double theta) {
  auto &communities = partition.getCommunities();

  for (auto const &node : subset.getNodes()) {
    if (isNodeWellConnected(node, subset, gamma)) {
      int index{static_cast<int>(partition.findCommunityIndex(node))};
      auto &nodeCommunity = communities[index];

      assert((communitySize(nodeCommunity)) != 0);
      if (communitySize(nodeCommunity) == 1) {
        int communityToIndex{
            extractRandomCommunityIndex(communities, partition, node, nodeCommunity, subset, gamma, theta)};
        auto &communityTo = communities[communityToIndex];
        moveNode(nodeCommunity, communityTo, node);
      }
    }
  }
  return partition;
}

Partition &refinePartition(
    TICLGraph const &graph, Partition &partition, Partition &singlePartition, double gamma, double theta) {
  //fills an empty partition with a singleton partition
  auto &refinedPartition = singletonPartition(graph, singlePartition);
  auto const &communities = partition.getCommunities();
  for (auto const &community : communities) {
    mergeNodesSubset(refinedPartition, community, gamma, theta);
  }
  return refinedPartition;
}

//is it ok to return this as a copy? or too expensive
void aggregateGraph(TICLGraph &graph, Partition const &partition) {
  //communities become nodes in aggregate graph
  std::vector<Community> const &communities{partition.getCommunities()};
  std::vector<Node> aggregatedNodes{};
  aggregatedNodes.reserve(communities.size());

  std::for_each(communities.begin(), communities.end(), [&aggregatedNodes](auto const &community) {
    aggregatedNodes.push_back(Node{community});
  });

  assert(aggregatedNodes.size() == communities.size());
  auto &oldNodes = graph.getNodes();
  oldNodes = aggregatedNodes;
}