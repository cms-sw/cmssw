#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TICLGraph.h"

void Node::findSubComponents(std::vector<Node>& graph, std::vector<unsigned int>& subComponent, std::string tabs) {
  tabs += "\t";
  if (!alreadyVisited_) {
    LogDebug("TICLGraph") << tabs << " Visiting node " << index_ << std::endl;
    alreadyVisited_ = true;
    subComponent.push_back(index_);
    for (auto const& neighbour : neighboursId_) {
      LogDebug("TICLGraph") << tabs << " Trying to visit " << neighbour << std::endl;
      graph[neighbour].findSubComponents(graph, subComponent, tabs);
    }
  }
}

std::vector<std::vector<unsigned int>> TICLGraph::findSubComponents() {
  std::vector<std::vector<unsigned int>> components;
  for (auto const& node : nodes_) {
    auto const id = node.getId();
    if (isRootNode_[id]) {
      //LogDebug("TICLGraph") << "DFS Starting From " << id << std::endl;
      std::string tabs = "\t";
      std::vector<unsigned int> tmpSubComponents;
      nodes_[id].findSubComponents(nodes_, tmpSubComponents, tabs);
      components.push_back(tmpSubComponents);
    }
  }
  return components;
}

void TICLGraph::dfsForCC(unsigned int nodeIndex,
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

std::vector<std::vector<unsigned int>> TICLGraph::getConnectedComponents() const {
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
