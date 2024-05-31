#ifndef RecoHGCal_TICL_TICLGraph_h
#define RecoHGCal_TICL_TICLGraph_h

#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include <unordered_set>
#include <cassert>
#include <algorithm>
#include <variant>
#include <vector>

//Defines classes Elementary, Community and variant Node, defines classes TICLGraph and Partition.
// an elementary node is a single trackster
class TICLGraph;

class Elementary {
  unsigned index_;
  bool isTrackster_;
  std::vector<unsigned int> outerNeighboursId_;
  std::vector<unsigned int> innerNeighboursId_;
  std::vector<unsigned int> neighboursId_;

  bool alreadyVisited_{false};
  //bool areCompatible(const std::vector<Node>& graph, const unsigned int& outerNode) { return true; };
  int degree_{0};

public:
  Elementary() = default;
  Elementary(unsigned index, bool isTrackster = true) : index_{index}, isTrackster_{isTrackster} {}

  Elementary(Elementary const&) = default;
  Elementary& operator=(Elementary const&) = default;
  Elementary(Elementary&&) = default;
  Elementary& operator=(Elementary&&) = default;
  ~Elementary() = default;

  void addOuterNeighbour(unsigned int trackster_id) { outerNeighboursId_.push_back(trackster_id); }
  void addInnerNeighbour(unsigned int trackster_id) { innerNeighboursId_.push_back(trackster_id); }
  void addNeighbour(unsigned int trackster_id) { neighboursId_.push_back(trackster_id); }
  unsigned int getId() const { return index_; }
  std::vector<unsigned int> const& getOuterNeighbours() const { return outerNeighboursId_; }
  std::vector<unsigned int> const& getInnerNeighbours() const { return innerNeighboursId_; }
  std::vector<unsigned int> const& getNeighbours() const { return neighboursId_; }
  int getDegree() const { return degree_; }

  void findSubComponents(std::vector<Elementary>& graph, std::vector<unsigned int>& subComponent, std::string tabs);

  bool isInnerNeighbour(const unsigned int tid) {
    auto findInner = std::find(innerNeighboursId_.begin(), innerNeighboursId_.end(), tid);
    return findInner != innerNeighboursId_.end();
  }
};

class Community;

// a node can consist of an elementary or of a community (needed to implement the aggregate graph)
using Node = std::variant<Elementary, Community>;

class Community {
  std::vector<Node> nodes_;
  int degree_;
  //degree is 1 if all Nodes are Elementary
  //degree is i+1 if all nodes are communities of degree i

public:
  Community(std::vector<Node> const& nodes, int degree) : nodes_{nodes}, degree_{degree} {}
  Community() = default;

  auto const& getNodes() const { return nodes_; }
  auto& getNodes() { return nodes_; }
  int getDegree() const { return degree_; }
  int getId() const { return 0; }
  void increaseDegree() { ++degree_; }
};

bool operator==(Node const& n1, Node const& n2);

struct Degree {
  template <class T>
  int operator()(T const& e) const {
    return e.getDegree();
  }
};

struct Id {
  template <class T>
  int operator()(T const& e) const {
    return e.getId();
  }
};

inline auto degree(Node const& node) { return std::visit(Degree{}, node); }

inline auto getId(Node const& node) { return std::visit(Id{}, node); }

int communitySize(Community const& community, int size = 0);

using Flat = std::vector<Elementary>;

void flatten(Community const& community, Flat& flat);

Flat flatten(Community const& community);

class TICLGraph {
  std::vector<Node> nodes_{};
  std::vector<int> isRootNode_{};

public:
  // can i remove default constructor ?? edm::Wrapper problem
  // without default constructor i could initialize connectedComponents when building the Graph
  TICLGraph() = default;
  TICLGraph(std::vector<Node> const& nodes) : nodes_{nodes} {}
  TICLGraph(std::vector<Node> const& nodes, std::vector<int> isRootNode) : nodes_{nodes}, isRootNode_{isRootNode} {}
  std::vector<Node> const& getNodes() const { return nodes_; }
  std::vector<Node>& getNodes() { return nodes_; }
  Node const& getNode(int i) const { return nodes_[i]; }
  std::vector<std::vector<unsigned int>> findSubComponents();

  TICLGraph(TICLGraph const&) = default;
  TICLGraph& operator=(TICLGraph const&) = default;
  TICLGraph(TICLGraph&&) = default;
  TICLGraph& operator=(TICLGraph&&) = default;
  ~TICLGraph() = default;
};

std::vector<std::vector<unsigned int>> findSubComponents(std::vector<Elementary>& graphElemNodes);

int numberOfEdges(Community const& communityA, Community const& communityB);

bool areNeighbours(Node const& nodeA, Node const& nodeB);

bool isCommunityContained(Community community, Community const& subset);

class Partition {
  std::vector<Community> communities_{};

  //a node is always in a community from the beginning so it always returns something
  auto findCommunityImpl(Node const& node) const {
    auto it = std::find_if(communities_.begin(), communities_.end(), [&](auto const& community) {
      return std::find(community.getNodes().begin(), community.getNodes().end(), node) != community.getNodes().end();
    });
    assert(it != communities_.end());
    return it;
  }

public:
  Partition(std::vector<Community> const& communities) : communities_{communities} {}
  std::vector<Community> const& getCommunities() const { return communities_; }
  std::vector<Community>& getCommunities() { return communities_; }

  auto& flattenPartition(std::vector<Flat>& flattenedPartition) {
    flattenedPartition.reserve(communities_.size());
    std::transform(communities_.begin(),
                   communities_.end(),
                   std::back_inserter(flattenedPartition),
                   [this](auto const& community) { return flatten(community); });

    return flattenedPartition;
  }

  Community const& findCommunity(Node const& node) const { return *findCommunityImpl(node); }

  auto findCommunityIndex(Node const& node) const {
    return std::distance(communities_.begin(), findCommunityImpl(node));
  }
};

#endif
