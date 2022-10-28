#include "RecoEcal/EgammaCoreTools/interface/GraphMap.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include <iomanip>

using namespace reco;

GraphMap::GraphMap(uint nNodes) : nNodes_(nNodes) {
  // One entry for node in the edges_ list
  edgesIn_.resize(nNodes);
  edgesOut_.resize(nNodes);
}

void GraphMap::addNode(const uint index, const NodeCategory category) {
  nodesCategories_[category].push_back(index);
  nodesCount_[category] += 1;
}

void GraphMap::addNodes(const std::vector<uint> &indices, const std::vector<NodeCategory> &categories) {
  for (size_t i = 0; i < indices.size(); i++) {
    addNode(indices[i], categories[i]);
  }
}

void GraphMap::addEdge(const uint i, const uint j) {
  // The first index is the starting node of the outcoming edge.
  edgesOut_.at(i).push_back(j);
  edgesIn_.at(j).push_back(i);
  // Adding a connection in the adjacency matrix only in one direction
  adjMatrix_[{i, j}] = 1.;
}

void GraphMap::setAdjMatrix(const uint i, const uint j, const float score) { adjMatrix_[{i, j}] = score; };

void GraphMap::setAdjMatrixSym(const uint i, const uint j, const float score) {
  adjMatrix_[{i, j}] = score;
  adjMatrix_[{j, i}] = score;
};

const std::vector<uint> &GraphMap::getOutEdges(const uint i) const { return edgesOut_.at(i); };

const std::vector<uint> &GraphMap::getInEdges(const uint i) const { return edgesIn_.at(i); };

uint GraphMap::getAdjMatrix(const uint i, const uint j) const { return adjMatrix_.at({i, j}); };

std::vector<float> GraphMap::getAdjMatrixRow(const uint i) const {
  std::vector<float> out;
  for (const auto &j : getOutEdges(i)) {
    out.push_back(adjMatrix_.at({i, j}));
  }
  return out;
};

std::vector<float> GraphMap::getAdjMatrixCol(const uint j) const {
  std::vector<float> out;
  for (const auto &i : getInEdges(j)) {
    out.push_back(adjMatrix_.at({i, j}));
  }
  return out;
};

//=================================================
// Debugging info
void GraphMap::printGraphMap() {
  edm::LogVerbatim("GraphMap") << "OUT edges" << std::endl;
  uint seed = 0;
  for (const auto &s : edgesOut_) {
    edm::LogVerbatim("GraphMap") << "cl: " << seed << " --> ";
    for (const auto &e : s) {
      edm::LogVerbatim("GraphMap") << e << " (" << adjMatrix_[{seed, e}] << ") ";
    }
    edm::LogVerbatim("GraphMap") << std::endl;
    seed++;
  }
  edm::LogVerbatim("GraphMap") << std::endl << "IN edges" << std::endl;
  seed = 0;
  for (const auto &s : edgesIn_) {
    edm::LogVerbatim("GraphMap") << "cl: " << seed << " <-- ";
    for (const auto &e : s) {
      edm::LogVerbatim("GraphMap") << e << " (" << adjMatrix_[{e, seed}] << ") ";
    }
    edm::LogVerbatim("GraphMap") << std::endl;
    seed++;
  }
  edm::LogVerbatim("GraphMap") << std::endl << "AdjMatrix" << std::endl;
  for (const auto &s : nodesCategories_[NodeCategory::kSeed]) {
    for (size_t n = 0; n < nNodes_; n++) {
      edm::LogVerbatim("GraphMap") << std::setprecision(2) << adjMatrix_[{s, n}] << " ";
    }
    edm::LogVerbatim("GraphMap") << std::endl;
  }
}

//--------------------------------------------------------------
// Nodes collection algorithms
void GraphMap::collectNodes(GraphMap::CollectionStrategy strategy, float threshold) {
  // Clear any stored graph output
  graphOutput_.clear();

  if (strategy == GraphMap::CollectionStrategy::Cascade) {
    // Starting from the highest energy seed, collect all the nodes.
    // Other seeds collected by higher energy seeds are ignored
    collectCascading(threshold);
  } else if (strategy == GraphMap::CollectionStrategy::CollectAndMerge) {
    // First, for each simple node (no seed) keep only the edge with the highest score.
    // Then collect all the simple nodes around the seeds.
    // Edges between the seed are ignored.
    // Finally, starting from the first seed (highest pt), look for linked secondary seed
    // and if they pass the threshold, merge their noded.
    assignHighestScoreEdge();
    const auto &[seedsGraph, simpleNodesMap] = collectSeparately(threshold);
    mergeSubGraphs(threshold, seedsGraph, simpleNodesMap);
  } else if (strategy == GraphMap::CollectionStrategy::SeedsFirst) {
    // Like strategy D, but after solving the edges between the seeds,
    // the simple nodes edges are cleaned to keep only the highest score link.
    // Then proceed as strategy B.
    resolveSuperNodesEdges(threshold);
    assignHighestScoreEdge();
    collectCascading(threshold);
  } else if (strategy == GraphMap::CollectionStrategy::CascadeHighest) {
    // First, for each simple node keep only the edge with the highest score.
    // Then proceed as strategy A, from the first seed cascading to the others.
    // Secondary seeds that are linked,  are absorbed and ignored in the next iteration:
    // this implies that nodes connected to these seeds are lost.
    assignHighestScoreEdge();
    collectCascading(threshold);
  }
}

//----------------------------------------
// Implementation of single actions

void GraphMap::collectCascading(float threshold) {
  // Starting from the highest energy seed, collect all the nodes.
  // Other seeds collected by higher energy seeds are ignored
  const auto &seeds = nodesCategories_[NodeCategory::kSeed];
  // seeds are already included in order
  LogDebug("GraphMap") << "Cascading...";
  for (const auto &s : seeds) {
    LogTrace("GraphMap") << "seed: " << s;
    std::vector<uint> collectedNodes;
    // Check if the seed if still available
    if (adjMatrix_[{s, s}] < threshold)
      continue;
    // Loop on the out-coming edges
    for (const auto &out : edgesOut_[s]) {
      // Check the threshold for association
      if (adjMatrix_[{s, out}] >= threshold) {
        LogTrace("GraphMap") << "\tOut edge: " << s << " --> " << out;
        // Save the node
        collectedNodes.push_back(out);
        // Remove all incoming edges to the selected node
        // So that it cannot be taken from other SuperNodes
        for (const auto &out_in : edgesIn_[out]) {
          // There can be 4 cases:
          // 1) out == s, out_in can be an incoming edge from other seed: to be removed
          // 2) out == s, out_in==s (self-loop): zeroing will remove the current node from the available ones
          // 3) out == r, out_in==s (current link): keep this
          // 4) out == r, out_in==r (self-loop on other seeds): remove this making the other seed not accessible
          if (out != s && out_in == s)
            continue;  // No need to remove the edge we are using
          adjMatrix_[{out_in, out}] = 0.;
          LogTrace("GraphMap") << "\t\t Deleted edge: " << out << " <-- " << out_in;
        }
      }
    }
    graphOutput_.push_back({s, collectedNodes});
  }
}

void GraphMap::assignHighestScoreEdge() {
  // First for each simple node (no seed) keep only the highest score link
  // Then perform strategy A.
  LogTrace("GraphMap") << "Keep only highest score edge";
  for (const auto &cl : nodesCategories_[NodeCategory::kNode]) {
    std::pair<uint, float> maxPair{0, 0};
    bool found = false;
    for (const auto &seed : edgesIn_[cl]) {
      float score = adjMatrix_[{seed, cl}];
      if (score > maxPair.second) {
        maxPair = {seed, score};
        found = true;
      }
    }
    if (!found)
      continue;
    LogTrace("GraphMap") << "cluster: " << cl << " edge from " << maxPair.first;
    // Second loop to remove all the edges apart from the max
    for (const auto &seed : edgesIn_[cl]) {
      if (seed != maxPair.first) {
        adjMatrix_[{seed, cl}] = 0.;
      }
    }
  }
}

std::pair<GraphMap::GraphOutput, GraphMap::GraphOutputMap> GraphMap::collectSeparately(float threshold) {
  // Save a subgraph of only seeds, without self-loops
  GraphOutput seedsGraph;
  // Collect all the nodes around seeds, but not other seeds
  GraphOutputMap simpleNodesGraphMap;
  LogDebug("GraphMap") << "Collecting separately each seed...";
  // seeds are already included in order
  for (const auto &s : nodesCategories_[NodeCategory::kSeed]) {
    LogTrace("GraphMap") << "seed: " << s;
    std::vector<uint> collectedNodes;
    std::vector<uint> collectedSeeds;
    // Check if the seed if still available
    if (adjMatrix_[{s, s}] < threshold)
      continue;
    // Loop on the out-coming edges
    for (const auto &out : edgesOut_[s]) {
      // Check if it is another seed
      // if out is a seed adjMatrix[self-loop] > 0
      if (out != s && adjMatrix_[{out, out}] > 0) {
        // DO NOT CHECK the score of the edge, it will be checked during the merging
        collectedSeeds.push_back(out);
        // No self-loops are saved in the seed graph output
        // Then continue and do not work on this edgeOut
        continue;
      }
      // Check the threshold for association
      if (adjMatrix_[{s, out}] >= threshold) {
        LogTrace("GraphMap") << "\tOut edge: " << s << " --> " << out << " (" << adjMatrix_[{s, out}] << " )";
        // Save the node
        collectedNodes.push_back(out);
        // The links of the node to other seeds are not touched
        // IF the function is called after assignHighestScoreEdge
        // the other links have been already removed.
        // IF not: the same node can be assigned to more subgraphs.
        // The self-loop is included in this case in order to save the seed node
        // in its own sub-graph.
      }
    }
    simpleNodesGraphMap[s] = collectedNodes;
    seedsGraph.push_back({s, collectedSeeds});
  }
  return std::make_pair(seedsGraph, simpleNodesGraphMap);
}

void GraphMap::mergeSubGraphs(float threshold, GraphOutput seedsGraph, GraphOutputMap nodesGraphMap) {
  // We have the graph between the seed and a map of
  // simple nodes connected to each seed separately.
  // Now we link them and build superGraphs starting from the first seed
  LogTrace("GraphMap") << "Starting merging";
  for (const auto &[s, other_seeds] : seedsGraph) {
    LogTrace("GraphMap") << "seed: " << s;
    // Check if the seed is still available
    if (adjMatrix_[{s, s}] < threshold)
      continue;
    // If it is, we collect the final list of nodes
    std::vector<uint> collectedNodes;
    // Take the previously connected simple nodes to the current seed one
    const auto &simpleNodes = nodesGraphMap[s];
    collectedNodes.insert(std::end(collectedNodes), std::begin(simpleNodes), std::end(simpleNodes));
    // Check connected seeds
    for (const auto &out_s : other_seeds) {
      // Check the score of the edge for the merging
      // and check if the other seed has not been taken already
      if (adjMatrix_[{out_s, out_s}] > threshold && adjMatrix_[{s, out_s}] > threshold) {
        LogTrace("GraphMap") << "\tMerging nodes from seed: " << out_s;
        // Take the nodes already linked to this seed
        const auto &otherNodes = nodesGraphMap[out_s];
        // We don't check for duplicates because assignHighestScoreEdge() should
        // have been called already
        collectedNodes.insert(std::end(collectedNodes), std::begin(otherNodes), std::end(otherNodes));
        // Now let's disable the secondary seed
        adjMatrix_[{out_s, out_s}] = 0.;
        // This makes the strategy  NOT RECURSIVE
        // Other seeds linked to the disable seed won't be collected, but analyzed independently.
      }
    }
    // Now remove the current seed from the available ones,
    // if not other seeds could take it and we would have a double use of objects.
    adjMatrix_[{s, s}] = 0;
    graphOutput_.push_back({s, collectedNodes});
  }
}

void GraphMap::resolveSuperNodesEdges(float threshold) {
  LogTrace("GraphMap") << "Resolving seeds";
  for (const auto &s : nodesCategories_[NodeCategory::kSeed]) {
    LogTrace("GraphMap") << "seed: " << s;
    // Check if the seed if still available
    if (adjMatrix_[{s, s}] < threshold)
      continue;
    // Loop on the out-coming edges
    for (const auto &out : edgesOut_[s]) {
      if (out != s && adjMatrix_[{out, out}] > 0 && adjMatrix_[{s, out}] > threshold) {
        // This is a link to another still available seed
        // If the edge score is good the other seed node is not disabled --> remove it
        LogTrace("GraphMap") << "\tdisable seed: " << out;
        adjMatrix_[{out, out}] = 0.;
        // Remove the edges from that seed
        // This is needed in order to be able to use the
        // assignHighestScoreEdge after this function correctly
        for (const auto &c : edgesOut_[out]) {
          adjMatrix_[{out, c}] = 0.;
          // We don't touch the edgesIn and edgesOut collections but we zero the adjmatrix --> more efficient
        }
      }
    }
  }
}
