#ifndef RecoEcal_EgammaCoreTools_GraphMap_h
#define RecoEcal_EgammaCoreTools_GraphMap_h

#include <vector>
#include <array>
#include <map>
#include <algorithm>

/*
 * Class handling a sparse graph of clusters.
 * 
 * Author: D. Valsecchi
 * Date:  08-02-2022
 */

namespace reco {

  class GraphMap {
  public:
    GraphMap(uint nNodes, const std::vector<uint> &categories);
    ~GraphMap(){};

    void printGraphMap();
    void addNode(const uint index, const uint category);
    void addNodes(const std::vector<uint> &indices, const std::vector<uint> &categories);
    void addEdge(const uint i, const uint j);
    void setAdjMatrix(const uint i, const uint j, const float score);
    void setAdjMatrixSym(const uint i, const uint j, const float score);

    //Getters
    const std::vector<uint> &getOutEdges(const uint i) const;
    const std::vector<uint> &getInEdges(const uint i) const;
    uint getAdjMatrix(const uint &i, const uint j) const;
    std::vector<float> getAdjMatrixRow(const uint i) const;
    std::vector<float> getAdjMatrixCol(const uint j) const;

    enum CollectionStrategy {
      A,  // Starting from the highest energy seed (cat1), collect all the nodes.
          // Other seeds collected by higher energy seeds (cat1) are ignored
      B,  // First, for each cat0 node keep only the edge with the highest score.
          // Then collect all the cat0 nodes around the cat1 seeds.
          // Edges between the cat1 nodes are ignored.
          // Finally, starting from the first cat1 node, look for linked cat1 secondary
          // nodes and if they pass the threshold, merge their noded.
      C,  // Like strategy D, but after solving the edges between the cat1 seeds,
          // the cat0 nodes edges are cleaned to keep only the highest score link.
          // Then proceed as strategy B.
      D   // First, for each cat0 node keep only the edge with the highest score.
      // Then proceed as strategy A, from the first cat1 node cascading to the others.
      // Secondary cat1 nodes linked are absorbed and ignored in the next iteration:
      // this implies that nodes connected to these cat1 nodes are lost.
    };

    // Output of the collection  [{seed, [list of clusters]}]
    typedef std::vector<std::pair<uint, std::vector<uint>>> GraphOutput;
    typedef std::map<uint, std::vector<uint>> GraphOutputMap;
    // Apply the collection algorithms
    const GraphOutput &collectNodes(const GraphMap::CollectionStrategy strategy, const float threshold);

  private:
    uint nNodes_;
    // Map with list of indices of nodes for each category
    std::map<uint, std::vector<uint>> nodesCategories_;
    // Count of nodes for each category
    std::map<uint, uint> nodesCount_;
    // Incoming edges, one list for each node (no distinction between type)
    std::vector<std::vector<uint>> edgesIn_;
    // Outcoming edges, one list for each node
    std::vector<std::vector<uint>> edgesOut_;
    // Adjacency matrix (i,j) --> score
    // Rows are interpreted as OUT edges
    // Columns are interpreted as IN edges
    std::map<std::pair<uint, uint>, float> adjMatrix_;

    // Store for the graph collection result
    GraphOutput graphOutput_;

    // Functions for the collection strategies
    void collectCascading(const float threshold);
    void assignHighestScoreEdge();
    // Return both the output graph with only cat1 nodes and a GraphOutputMap
    // of the collected cat0 nodes from each cat1 one.
    std::pair<GraphOutput, GraphOutputMap> collectSeparately(const float threshold);
    void mergeSubGraphs(const float threshold, const GraphOutput &cat1NodesGraph, const GraphOutputMap &cat0GraphMap);
    void resolveSuperNodesEdges(const float threshold);
  };

}  // namespace reco

#endif
