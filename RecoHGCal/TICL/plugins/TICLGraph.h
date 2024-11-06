#ifndef DataFormats_HGCalReco_TICLGraph_h
#define DataFormats_HGCalReco_TICLGraph_h

#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include <unordered_set>

namespace ticl {

  class Node {
  public:
    Node() = default;
    Node(unsigned index, bool isTrackster = true) : index_(index), isTrackster_(isTrackster), alreadyVisited_{false} {};

    inline void addOuterNeighbour(unsigned int trackster_id) { outerNeighboursId_.push_back(trackster_id); }
    inline void addInnerNeighbour(unsigned int trackster_id) { innerNeighboursId_.push_back(trackster_id); }

    inline const unsigned int getId() const { return index_; }
    const std::vector<unsigned int>& getOuterNeighbours() const { return outerNeighboursId_; }
    const std::vector<unsigned int>& getInnerNeighbours() const { return innerNeighboursId_; }
    void findSubComponents(std::vector<Node>& graph, std::vector<unsigned int>& subComponent);

    inline bool isInnerNeighbour(const unsigned int tid) {
      auto findInner = std::find(innerNeighboursId_.begin(), innerNeighboursId_.end(), tid);
      return findInner != innerNeighboursId_.end();
    }
    inline bool alreadyVisited() const { return alreadyVisited_; }

    ~Node() = default;

  private:
    unsigned int index_;
    bool isTrackster_;

    std::vector<unsigned int> outerNeighboursId_;
    std::vector<unsigned int> innerNeighboursId_;
    bool alreadyVisited_;

    //bool areCompatible(const std::vector<Node>& graph, const unsigned int& outerNode) { return true; };
  };
}  // namespace ticl

class TICLGraph {
public:
  // can i remove default constructor ?? edm::Wrapper problem
  // without default constructor i could initialize connectedComponents when building the Graph
  TICLGraph() = default;
  TICLGraph(std::vector<ticl::Node>& nodes);
  inline const std::vector<ticl::Node>& getNodes() const { return nodes_; }
  inline const ticl::Node& getNode(int i) const { return nodes_[i]; }
  inline std::vector<ticl::Node> getRootNodes() const { return rootNodes_; }
  inline void findRootNodes();

  std::vector<std::vector<unsigned int>> findSubComponents();
  std::vector<std::vector<unsigned int>> findSubComponents(std::vector<ticl::Node>& rootNodes);

  ~TICLGraph() = default;

  std::vector<std::vector<unsigned int>> getConnectedComponents() const;
  bool isGraphOk();

private:
  std::vector<ticl::Node> nodes_;
  std::vector<ticl::Node> rootNodes_;
  std::vector<int> isRootNode_;
};

#endif
