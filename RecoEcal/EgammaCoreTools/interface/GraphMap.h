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
    GraphMap(uint nNodes);

    enum NodeCategory { kNode, kSeed, kNcategories };

    void addNode(const uint index, const NodeCategory category);
    void addNodes(const std::vector<uint> &indices, const std::vector<NodeCategory> &categories);
    void addEdge(const uint i, const uint j);
    void setAdjMatrix(const uint i, const uint j, const float score);
    void setAdjMatrixSym(const uint i, const uint j, const float score);
    void printGraphMap();

    //Getters
    const std::vector<uint> &getOutEdges(const uint i) const;
    const std::vector<uint> &getInEdges(const uint i) const;
    uint getAdjMatrix(const uint i, const uint j) const;
    std::vector<float> getAdjMatrixRow(const uint i) const;
    std::vector<float> getAdjMatrixCol(const uint j) const;

    enum CollectionStrategy {
      Cascade,          // Starting from the highest energy seed, collect all the nodes.
                        // Other seeds collected by higher energy seeds are ignored
      CollectAndMerge,  // First, for each simple node keep only the edge with the highest score.
                        // Then collect all the simple nodes around the other seeds.
                        // Edges between the seeds nodes are ignored.
                        // Finally, starting from the first seed, look for linked secondary seeds
                        // and if they pass the threshold, merge their noded.
      SeedsFirst,       // Like strategy D, but after solving the edges between the seeds,
                        // the simple nodes edges are cleaned to keep only the highest score link.
                        // Then proceed as strategy B.
      CascadeHighest    // First, for each simple node keep only the edge with the highest score.
      // Then proceed as strategy A, from the first seed node cascading to the others.
      // Secondary seeds linked are absorbed and ignored in the next iteration:
      // this implies that nodes connected to these seed  are lost.
    };

    // Output of the collection  [{seed, [list of clusters]}]
    typedef std::vector<std::pair<uint, std::vector<uint>>> GraphOutput;
    typedef std::map<uint, std::vector<uint>> GraphOutputMap;
    // Apply the collection algorithms
    void collectNodes(GraphMap::CollectionStrategy strategy, float threshold);
    const GraphOutput &getGraphOutput() { return graphOutput_; };

  private:
    uint nNodes_;
    // Map with list of indices of nodes for each category
    std::map<NodeCategory, std::vector<uint>> nodesCategories_;
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
    void collectCascading(float threshold);
    void assignHighestScoreEdge();
    // Return both the output graph with only seedss and a GraphOutputMap
    // of the collected simple nodes from each seed.
    std::pair<GraphOutput, GraphOutputMap> collectSeparately(float threshold);
    void mergeSubGraphs(float threshold, GraphOutput seedsGraph, GraphOutputMap nodesGraphMap);
    void resolveSuperNodesEdges(float threshold);
  };

}  // namespace reco

#endif
