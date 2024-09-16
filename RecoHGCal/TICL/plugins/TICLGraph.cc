#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TICLGraph.h"

namespace ticl {

  void Node::findSubComponents(std::vector<Node>& graph, std::vector<unsigned int>& subComponent, std::string tabs) {
    tabs += "\t";
    if (!alreadyVisited_) {
      LogDebug("TICLGraph") << tabs << " Visiting node " << index_ << std::endl;
      alreadyVisited_ = true;
      subComponent.push_back(index_);
      for (auto const& neighbour : outerNeighboursId_) {
        LogDebug("TICLGraph") << tabs << " Trying to visit " << neighbour << std::endl;
        graph[neighbour].findSubComponents(graph, subComponent, tabs);
      }
    }
  }
}  // namespace ticl

TICLGraph::TICLGraph(std::vector<ticl::Node>& nodes) {
  nodes_ = nodes;
  rootNodes_.reserve(nodes_.size());
  findRootNodes();
  rootNodes_.shrink_to_fit();
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
  // Second loop: DFS for non-root nodes that haven't been visited
  for (auto const& node : nodes_) {
    auto const id = node.getId();
    if (!node.alreadyVisited()) {  // Use the alreadyVisited() method
      // Node not visited yet, so perform DFS
      std::string tabs = "\t";
      std::vector<unsigned int> tmpSubComponents;
      nodes_[id].findSubComponents(nodes_, tmpSubComponents, tabs);
      components.push_back(tmpSubComponents);
    }
  }
  return components;
}

std::vector<std::vector<unsigned int>> TICLGraph::findSubComponents(std::vector<ticl::Node>& rootNodes) {
  std::vector<std::vector<unsigned int>> components;
  for (auto const& node : rootNodes) {
    auto const id = node.getId();
    //LogDebug("TICLGraph") << "DFS Starting From " << id << std::endl;
    std::string tabs = "\t";
    std::vector<unsigned int> tmpSubComponents;
    nodes_[id].findSubComponents(nodes_, tmpSubComponents, tabs);
    components.push_back(tmpSubComponents);
  }
  return components;
}

inline void TICLGraph::findRootNodes() {
  for (auto const& n : nodes_) {
    if (n.getInnerNeighbours().size() == 0) {
      rootNodes_.push_back(n);
    }
  }
}

bool TICLGraph::isGraphOk() {
  for (auto n : nodes_) {
    if (n.getInnerNeighbours().size() > 1) {
      return false;
    }
  }
  return true;
}
