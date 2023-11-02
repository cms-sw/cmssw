#ifndef DataFormats_HGCalReco_TICLGraph_h
#define DataFormats_HGCalReco_TICLGraph_h

#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include <unordered_set>

class Node {
public:
  Node() = default;
  Node(unsigned index, bool isTrackster = true) : index_(index), isTrackster_(isTrackster), alreadyVisited_{false}{};

  void addNeighbour(unsigned int trackster_id) {
    neighboursId_.push_back(trackster_id);
  }

  const unsigned int getId() const { return index_; }
  std::vector<unsigned int> getNeighbours() const { return neighboursId_; }
  void findSubComponents(std::vector<Node>& graph, std::vector<unsigned int>& subComponent, std::string tabs) {
    tabs += "\t";
    if (!alreadyVisited_) {
//      std::cout << tabs << " Visiting node " << index_ << std::endl;
      alreadyVisited_ = true;
      subComponent.push_back(index_);

      for (auto const& neighbour : neighboursId_) {
        //std::cout << tabs << " Trying to visit " << neighbour << std::endl;
        graph[neighbour].findSubComponents(graph, subComponent, tabs);
      }
    }
  }

  ~Node() = default;

private:
  unsigned index_;
  bool isTrackster_;

  std::vector<unsigned int> neighboursId_;
  bool alreadyVisited_;

  //bool areCompatible(const std::vector<Node>& graph, const unsigned int& outerNode) { return true; };

};

class TICLGraph {
public:
  // can i remove default constructor ?? edm::Wrapper problem
  // without default constructor i could initialize connectedComponents when building the Graph
  TICLGraph() = default;
  TICLGraph(std::vector<Node>& n, std::vector<int> isRootNode) {
    nodes_ = n;
//    isRootNode.resize(nodes_.size());
    isRootNode_ = isRootNode;
  };
  const std::vector<Node>& getNodes() const { return nodes_; }
  const Node& getNode(int i) const { return nodes_[i]; }

//  void setRootNodes() {
//    for (auto const& node : nodes_) {
//      bool isRootNode = condition ? true : false;
//      rootNodeIds[node.getId()] = isRootNode;
//    }
//  }

  std::vector<std::vector<unsigned int>> findSubComponents() {
    std::vector<std::vector<unsigned int>> components;
    for (auto const& node: nodes_) {
      auto const id = node.getId();
      if(isRootNode_[id]){
        //std::cout << "DFS Starting From " << id << std::endl;
        std::string tabs = "\t";
        std::vector<unsigned int> tmpSubComponents;
        nodes_[id].findSubComponents(nodes_, tmpSubComponents, tabs);
        components.push_back(tmpSubComponents);
      }
    }
    return components;
  }

  ~TICLGraph() = default;

  void dfsForCC(unsigned int nodeIndex,
                std::unordered_set<unsigned int>& visited,
                std::vector<unsigned int>& component) const {
    visited.insert(nodeIndex);
    component.push_back(nodeIndex);

    for (auto const& neighbourIndex : nodes_[nodeIndex].getNeighbours()) {
      if (visited.find(neighbourIndex) == visited.end()) {
        dfsForCC(neighbourIndex, visited, component);
      }
    }
  }

  std::vector<std::vector<unsigned int>> getConnectedComponents() const {
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
  }

private:
  std::vector<Node> nodes_;
  std::vector<int> isRootNode_;
};

#endif
