#include "DataFormats/HGCalReco/interface/TICLGraph.h"

bool operator==(ElementaryNode const& eN1, ElementaryNode const& eN2) {
  return (eN1.getId() == eN2.getId()) && (eN1.getNeighbours() == eN2.getNeighbours());
}

template <class T>
bool operator==(Node<T> const& n1, Node<T> const& n2) {
  return ((n1.getInternalStructure()) == (n2.getInternalStructure()));
}

//fills an empty vector with the result of flattening operation of a community
template <class T>
std::vector<ElementaryNode>& flatCommunity(std::vector<Node<T>> const& community,
                                           std::vector<ElementaryNode>& flattenedCommunity) {
  assert(flattenedCommunity.empty());
  for (auto const& node : community) {
    if (node.isNodeDegreeZero()) {
      for (auto const& elementaryNode : node.getInternalStructure())
        flattenedCommunity.push_back(elementaryNode);
    } else
      flatCommunity(node.getInternalStructure(), flattenedCommunity);
  }
  return flattenedCommunity;
}

// the number of edges b/w 2 nodes is the number of edges between their elementary nodes
template <class T>
int numberOfEdges(std::vector<Node<T>> const& communityA, std::vector<Node<T>> const& communityB) {
  int numberOfEdges{};

  std::vector<ElementaryNode> flattenedCommunityA{};
  std::vector<ElementaryNode> flattenedCommunityB{};
  flatCommunity(communityA, flattenedCommunityA);
  flatCommunity(communityB, flattenedCommunityB);

  for (auto const& elementaryNodeA : flattenedCommunityA) {
    std::vector<unsigned int> const& neighboursA{elementaryNodeA.getNeighbours()};
    for (auto const& Id : neighboursA) {
      auto it{std::find_if(flattenedCommunityB.begin(),
                           flattenedCommunityB.end(),
                           [=](ElementaryNode const& elNodeB) { return (elNodeB.getId()) == Id; })};
      if (it != flattenedCommunityB.end()) {
        ++numberOfEdges;
      }
    }
  }
  assert(numberOfEdges >= 0);
  return numberOfEdges;
}

//the size of a community is the number of elementary nodes in it
template <class T>
int communitySize(std::vector<Node<T>> const& community, int size = 0) {
  for (auto const& node : community) {
    if (node.isNodeDegreeZero()) {
      size += node.size();
    } else
      size = communitySize(node.getInternalStructure(), size);
  }
  return size;
}

template <class T>
bool areNeighbours(Node<T> const& nodeA, Node<T> const& nodeB) {
  std::vector<Node<T>> A{nodeA};
  std::vector<Node<T>> B{nodeB};
  std::vector<ElementaryNode> flattenedCommunityA{};
  std::vector<ElementaryNode> flattenedCommunityB{};
  flatCommunity(A, flattenedCommunityA);
  flatCommunity(B, flattenedCommunityB);
  bool result{false};
  for (auto const& elementaryNodeA : flattenedCommunityA) {
    std::vector<unsigned int> const& neighboursA{elementaryNodeA.getNeighbours()};
    for (auto const& Id : neighboursA) {
      auto it{std::find_if(flattenedCommunityB.begin(),
                           flattenedCommunityB.end(),
                           [&Id](ElementaryNode const& elNodeB) { return (elNodeB.getId()) == Id; })};
      //for two nodes to be neighbours i simply need two of their elementary nodes being neighbours
      if (it != flattenedCommunityB.end()) {
        result = true;
        break;
      }
    }
  }
  return result;
}

template <class T>
//tells me if community is contained within a certain subset
bool isCommunityContained(std::vector<Node<T>> const& community, std::vector<Node<T>> const& subset) {
  bool isContained{true};
  for (auto const& node : community) {
    auto it{std::find(subset.begin(), subset.end(), node)};
    if (it == subset.end()) {
      isContained = false;
      break;
    }
  }
  return isContained;
}