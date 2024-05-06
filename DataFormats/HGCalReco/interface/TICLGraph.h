#ifndef DataFormats_HGCalReco_TICLGraph_h
#define DataFormats_HGCalReco_TICLGraph_h

#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include <unordered_set>
#include <cassert>
#include <algorithm>

//NB NODE NEEDS AN ADDNEIGHBOUR METHOD BC TICLGRAPHPRODUCER NEEDS IT

// an elementary node is a single trackster
class ElementaryNode {
  unsigned index_;
  bool isTrackster_;
  std::vector<unsigned int> neighboursId_{};
  bool alreadyVisited_{false};
  //bool areCompatible(const std::vector<Node>& graph, const unsigned int& outerNode) { return true; };

public:
  ElementaryNode() = default;
  ElementaryNode(unsigned index, bool isTrackster = true) : index_(index), isTrackster_(isTrackster) {}

  //can i remove default dctor so i dont have to apply rule of 5???
  ElementaryNode(ElementaryNode const&) = default;
  ElementaryNode& operator=(ElementaryNode const&) = default;
  ElementaryNode(ElementaryNode&&) = default;
  ElementaryNode& operator=(ElementaryNode&&) = default;
  ~ElementaryNode() = default;

  void addNeighbour(unsigned int trackster_id) { neighboursId_.push_back(trackster_id); }
  const unsigned int getId() const { return index_; }
  std::vector<unsigned int> const& getNeighbours() const { return neighboursId_; }
  int size() const { return 0; }
  /* void findSubComponents(std::vector<Node>& graph, std::vector<unsigned int>& subComponent, std::string tabs) {
    tabs += "\t";
    if (!alreadyVisited_) {
//  std::cout << tabs << " Visiting node " << index_ << std::endl;
    alreadyVisited_ = true;
    subComponent.push_back(index_);

    for (auto const& neighbour : neighboursId_) {
       //std::cout << tabs << " Trying to visit " << neighbour << std::endl;
        graph[neighbour].findSubComponents(graph, subComponent, tabs);
      }
    }
  }*/
};

bool operator==(ElementaryNode const& eN1, ElementaryNode const& eN2);

// a node can contain one or more elementary nodes (needed to implement the aggregate graph)
template <class T>
class Node {
  std::vector<T> internalStructure_{};

public:
  Node() = default;
  Node(std::vector<T> const& internalStructure) : internalStructure_{internalStructure} {
    assert(internalStructure.size() != 0);
  };
  const std::vector<T>& getInternalStructure() const { return internalStructure_; }
  //THIS implies that NO node can be a vector of size zero. Therefore, before creating the aggregate graph, one should remove empty communities.
  //a node is of degree zero if it consists in a vector of ElementaryNodes
  bool isNodeDegreeZero() const {
    assert(internalStructure_.size() != 0);
    return (internalStructure_[0].size() == 0) ? true : false;
  }
};

//tested this implementation on godbolt it should work
template <class T>
bool operator==(Node<T> const& n1, Node<T> const& n2);

template <class T>
class TICLGraph {
  std::vector<Node<T>> nodes_{};
  std::vector<int> isRootNode_{};

public:
  // can i remove default constructor ?? edm::Wrapper problem
  // without default constructor i could initialize connectedComponents when building the Graph
  TICLGraph() = default;
  TICLGraph(std::vector<Node<T>> const& nodes) : nodes_{nodes} {}
  TICLGraph(std::vector<Node<T>> const& nodes, std::vector<int> isRootNode) : nodes_{nodes}, isRootNode_{isRootNode} {}
  std::vector<Node<T>> const& getNodes() const { return nodes_; }
  Node<T> const& getNode(int i) const { return nodes_[i]; }

  //  void setRootNodes() {
  //    for (auto const& node : nodes_) {
  //      bool isRootNode = condition ? true : false;
  //      rootNodeIds[node.getId()] = isRootNode;
  //    }
  //  }

  /*std::vector<std::vector<unsigned int>> findSubComponents() {
    std::vector<std::vector<unsigned int>> components;
    for (auto const& node : nodes_) {
      auto const id = node.getId();
      if (isRootNode_[id]) {
        //std::cout << "DFS Starting From " << id << std::endl;
        std::string tabs = "\t";
        std::vector<unsigned int> tmpSubComponents;
        nodes_[id].findSubComponents(nodes_, tmpSubComponents, tabs);
        components.push_back(tmpSubComponents);
      }
    }
    return components;
  }*/

  TICLGraph(TICLGraph const&) = default;
  TICLGraph& operator=(TICLGraph const&) = default;
  TICLGraph(TICLGraph&&) = default;
  TICLGraph& operator=(TICLGraph&&) = default;
  ~TICLGraph() = default;

  /*void dfsForCC(unsigned int nodeIndex,
                std::unordered_set<unsigned int>& visited,
                std::vector<unsigned int>& component) const {
    visited.insert(nodeIndex);
    component.push_back(nodeIndex);

    for (auto const& neighbourIndex : nodes_[nodeIndex].getNeighbours()) {
      if (visited.find(neighbourIndex) == visited.end()) {
        dfsForCC(neighbourIndex, visited, component);
      }
    }
  }*/

  /*std::vector<std::vector<unsigned int>> getConnectedComponents() const {
    std::unordered_set<unsigned int> visited;
    std::vector<std::vector<unsigned int>> components;

    for (unsigned int i = 0; i < nodes_.size(); ++i) {
      if (visited.find(i) == visited.end()) {
        std::vector<unsigned int> component;
        dfsForCC(i, visited, component);
        components.push_back(component);
      }
    }

    return components;
  }*/
};

//takes a community and return the vector of all the Elementary Nodes in the community
template <class T>
auto flatCommunity(std::vector<Node<T>> const& community, std::vector<ElementaryNode>& flattenedCommunity);

template <class T>
class Partition {
  std::vector<std::vector<Node<T>>> communities_{};

public:
  Partition(std::vector<std::vector<Node<T>>> communities) : communities_{communities} {}
  const std::vector<std::vector<Node<T>>>& getPartition() const { return communities_; }
  std::vector<std::vector<Node<T>>>& setPartition() { return communities_; }
  auto flatPartition(std::vector<std::vector<ElementaryNode>> const& flattenedPartition) {
    for (auto& community : communities_) {
      std::vector<ElementaryNode> flattenedCommunity{};
      flattenedPartition.push_back(flatCommunity(community, flattenedCommunity));
    }
    return flattenedPartition;
  }

  //implemented on the assumption that when I work with community of std::vector<Node<T>> my node will be Node<T> (nesting degree matches)
  //a node is always in a community from the beginning so it always returns something
  std::vector<Node<T>> const& findCommunity(Node<T> const& node) const {
    for (auto const& community : communities_) {
      auto it{std::find(community.begin(), community.end(), node)};
      if (it != community.end()) {
        return community;
      }
    }
  }
};

template <class T>
int numberOfEdges(std::vector<Node<T>> const& communityA, std::vector<Node<T>> const& communityB);

template <class T>
int communitySize(std::vector<Node<T>> const& community);

template <class T>
bool areNeighbours(Node<T> const& nodeA, Node<T> const& nodeB);

//tells me if community is contained within a certain subset
/*bool isCommunityContained (std::vector<Node<T>> const& community, std::vector<Node<T>> const& subset){
for (auto const& node : community) {
  std::find
}
}*/

#endif
