#include "DataFormats/HGCalReco/interface/TICLGraph.h"

struct Equal {
  bool operator()(Elementary const& e1, Elementary const& e2) {
    return (e1.getId() == e2.getId()) && (e1.getNeighbours() == e2.getNeighbours());
  }

  bool operator()(Community const& c1, Community const& c2) { return c1.getNodes() == c2.getNodes(); }

  //these two are necessary bc i need to cover all combinations. A node is never equal to a community by definition
  bool operator()(Community const& c1, Elementary const& e2) { return false; }
  bool operator()(Elementary const& e1, Community const& c2) { return false; }
};

bool operator==(Node const& n1, Node const& n2) { return std::visit(Equal{}, n1, n2); }

void flatten(Community const& community, Flat& flat) {
  for (auto& node : community.getNodes()) {
    if (auto* elementary = std::get_if<Elementary>(&node); elementary != nullptr) {
      flat.push_back(*elementary);
    } else {
      flatten(std::get<Community>(node), flat);
    }
  }
}

Flat flatten(Community const& community) {
  int const size{communitySize(community)};
  Flat flattenedCommunity{};
  flattenedCommunity.reserve(size);
  flatten(community, flattenedCommunity);
  return flattenedCommunity;
}

//godbolt gives error if i put =0 in definition instead of declaration
int communitySize(Community const& community, int size) {
  for (auto const& node : community.getNodes()) {
    if (std::holds_alternative<Elementary>(node)) {
      ++size;
    } else
      size = communitySize(std::get<Community>(node), size);
  }
  return size;
}

// the number of edges b/w 2 nodes is the number of edges between their elementary nodes
int numberOfEdges(Community const& communityA, Community const& communityB) {
  auto flattenedCommunityA = flatten(communityA);
  auto flattenedCommunityB = flatten(communityB);
  std::vector<unsigned int> membersB{};
  membersB.reserve(flattenedCommunityB.size());
  std::transform(flattenedCommunityB.begin(), flattenedCommunityB.begin(), std::back_inserter(membersB), [](auto& e) {
    return e.getId();
  });
  std::sort(membersB.begin(), membersB.end());

  int numberOfEdges{0};
  for (auto const& elementaryNodeA : flattenedCommunityA) {
    auto neighboursA = elementaryNodeA.getNeighbours();
    std::sort(neighboursA.begin(), neighboursA.end());
    auto it = std::set_intersection(
        neighboursA.begin(), neighboursA.end(), membersB.begin(), membersB.end(), neighboursA.begin());
    numberOfEdges += std::distance(neighboursA.begin(), it);
  }
  assert(numberOfEdges >= 0);
  return numberOfEdges;
}

struct Neighbours {
  bool operator()(Elementary const& e1, Elementary const& e2) {
    auto const& neighbours1 = e1.getNeighbours();
    auto id2 = e2.getId();
    return (std::find(neighbours1.begin(), neighbours1.end(), id2) != neighbours1.end());
  }

  bool operator()(Community const& c1, Community const& c2) {
    auto flattenedCommunity1 = flatten(c1);
    auto flattenedCommunity2 = flatten(c2);

    bool result{false};
    for (auto const& elementary1 : flattenedCommunity1) {
      std::vector<unsigned int> const& neighbours1{elementary1.getNeighbours()};
      for (auto const& Id : neighbours1) {
        auto it = std::find_if(flattenedCommunity2.begin(), flattenedCommunity2.end(), [=](Elementary const& e) {
          return (e.getId()) == Id;
        });
        if (it != flattenedCommunity2.end()) {
          result = true;
          break;
        }
      }
    }
    return result;
  }

  //these two are necessary bc i need to cover all combinations. A node cannot be neigbour of a community by definition
  bool operator()(Community const& c1, Elementary const& c2) { return false; }
  bool operator()(Elementary const& c1, Community const& c2) { return false; }
};

// two nodes are neighbours if at least two of their elementary nodes are
bool areNeighbours(Node const& n1, Node const& n2) { return std::visit(Neighbours{}, n1, n2); }

//tells me if community is contained within a certain subset
bool isCommunityContained(Community community, Community const& subset) {
  bool isContained{true};
  for (auto const& node : community.getNodes()) {
    auto it = std::find(subset.getNodes().begin(), subset.getNodes().end(), node);
    if (it == subset.getNodes().end()) {
      isContained = false;
      break;
    }
  }
  return isContained;
}