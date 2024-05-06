#include "DataFormats/HGCalReco/interface/TICLGraph.h"

bool operator==(ElementaryNode const& eN1, ElementaryNode const& eN2) {
  return (eN1.getId() == eN2.getId()) && (eN1.getNeighbours() == eN2.getNeighbours());
}

template <class T>
inline bool operator==(Node<T> const& n1, Node<T> const& n2) {
  return ((n1.getInternalStructure()) == (n2.getInternalStructure()));
}

template <class T>
auto flatCommunity(std::vector<Node<T>> const& community, std::vector<ElementaryNode>& flattenedCommunity) {
  for (auto const& node : community) {
    if (node.isNodeDegreeZero()) {
      for (auto const& elementaryNode : node.getInternalStructure())
        flattenedCommunity.push_back(elementaryNode);
    } else
      flatCommunity(node, flattenedCommunity);
  }
  return flattenedCommunity;
}

template <class T>
// the number of edges b/w 2 nodes is the number of edges between their elementary nodes
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
                           [&Id](ElementaryNode const& elNodeB) { return (elNodeB.getId()) == Id; })};
      if (it != flattenedCommunityB.end()) {
        ++numberOfEdges;
      }
    }
  }
  return numberOfEdges;
}

template <class T>
//the size of a community is the number of elementary nodes in it
int communitySize(std::vector<Node<T>> const& community) {
  int size{};
  for (auto const& node : community) {
    if (node.isNodeDegreeZero()) {
      size += node.size();
    } else
      communitySize(node);
  }
  return size;
}