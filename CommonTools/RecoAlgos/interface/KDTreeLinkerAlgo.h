#ifndef KDTreeLinkerAlgoTemplated_h
#define KDTreeLinkerAlgoTemplated_h

#include <cassert>
#include <vector>
#include <array>
#include <algorithm>

// Box structure used to define 2D field.
// It's used in KDTree building step to divide the detector
// space (ECAL, HCAL...) and in searching step to create a bounding
// box around the demanded point (Track collision point, PS projection...).
template <unsigned DIM = 2>
struct KDTreeBox {
  std::array<float, DIM> dimmin, dimmax;

  template <typename... Ts>
  KDTreeBox(Ts... dimargs) {
    static_assert(sizeof...(dimargs) == 2 * DIM, "Constructor requires 2*DIM args");
    std::vector<float> dims = {dimargs...};
    for (unsigned i = 0; i < DIM; ++i) {
      dimmin[i] = dims[2 * i];
      dimmax[i] = dims[2 * i + 1];
    }
  }

  KDTreeBox() {}
};

// Data stored in each KDTree node.
// The dim1/dim2 fields are usually the duplication of some PFRecHit values
// (eta/phi or x/y). But in some situations, phi field is shifted by +-2.Pi
template <typename DATA, unsigned DIM = 2>
struct KDTreeNodeInfo {
  DATA data;
  std::array<float, DIM> dims;

public:
  KDTreeNodeInfo() {}

  template <typename... Ts>
  KDTreeNodeInfo(const DATA &d, Ts... dimargs) : data(d), dims{{dimargs...}} {}
  template <typename... Ts>
  bool operator>(const KDTreeNodeInfo &rhs) const {
    return (data > rhs.data);
  }
};

template <typename DATA, unsigned DIM = 2>
struct KDTreeNodes {
  std::array<std::vector<float>, DIM> dims;
  std::vector<int> right;
  std::vector<DATA> data;

  int poolSize;
  int poolPos;

  constexpr KDTreeNodes() : poolSize(-1), poolPos(-1) {}

  bool empty() const { return poolPos == -1; }
  int size() const { return poolPos + 1; }

  void clear() {
    for (auto &dim : dims) {
      dim.clear();
    }
    right.clear();
    data.clear();
    poolSize = -1;
    poolPos = -1;
  }

  int getNextNode() {
    ++poolPos;
    return poolPos;
  }

  void build(int sizeData) {
    poolSize = sizeData * 2 - 1;
    for (auto &dim : dims) {
      dim.resize(poolSize);
    }
    right.resize(poolSize);
    data.resize(poolSize);
  };

  constexpr bool isLeaf(int right) const {
    // Valid values of right are always >= 2
    // index 0 is the root, and 1 is the first left node
    // Exploit index values 0 and 1 to mark which of dim1/dim2 is the
    // current one in recSearch() at the depth of the leaf.
    return right < 2;
  }

  bool isLeafIndex(int index) const { return isLeaf(right[index]); }
};

// Class that implements the KDTree partition of 2D space and
// a closest point search algorithme.

template <typename DATA, unsigned int DIM = 2>
class KDTreeLinkerAlgo {
public:
  // Dtor calls clear()
  ~KDTreeLinkerAlgo() { clear(); }

  // Here we build the KD tree from the "eltList" in the space define by "region".
  void build(std::vector<KDTreeNodeInfo<DATA, DIM> > &eltList, const KDTreeBox<DIM> &region);

  // Here we search in the KDTree for all points that would be
  // contained in the given searchbox. The founded points are stored in resRecHitList.
  void search(const KDTreeBox<DIM> &searchBox, std::vector<DATA> &resRecHitList);

  // This reurns true if the tree is empty
  bool empty() { return nodePool_.empty(); }

  // This returns the number of nodes + leaves in the tree
  // (nElements should be (size() +1)/2)
  int size() { return nodePool_.size(); }

  // This method clears all allocated structures.
  void clear() { clearTree(); }

private:
  // The node pool allow us to do just 1 call to new for each tree building.
  KDTreeNodes<DATA, DIM> nodePool_;

  std::vector<DATA> *closestNeighbour;
  std::vector<KDTreeNodeInfo<DATA, DIM> > *initialEltList;

  //Fast median search with Wirth algorithm in eltList between low and high indexes.
  int medianSearch(int low, int high, int treeDepth) const;

  // Recursif kdtree builder. Is called by build()
  int recBuild(int low, int hight, int depth);

  // Recursif kdtree search. Is called by search()
  void recSearch(int current, const KDTreeBox<DIM> &trackBox, int depth = 0) const;

  // This method frees the KDTree.
  void clearTree() { nodePool_.clear(); }
};

//Implementation

template <typename DATA, unsigned int DIM>
void KDTreeLinkerAlgo<DATA, DIM>::build(std::vector<KDTreeNodeInfo<DATA, DIM> > &eltList,
                                        const KDTreeBox<DIM> &region) {
  if (!eltList.empty()) {
    initialEltList = &eltList;

    size_t size = initialEltList->size();
    nodePool_.build(size);

    // Here we build the KDTree
    int root = recBuild(0, size, 0);
    assert(root == 0);

    initialEltList = nullptr;
  }
}

//Fast median search with Wirth algorithm in eltList between low and high indexes.
template <typename DATA, unsigned int DIM>
int KDTreeLinkerAlgo<DATA, DIM>::medianSearch(int low, int high, int treeDepth) const {
  int nbrElts = high - low;
  int median = (nbrElts & 1) ? nbrElts / 2 : nbrElts / 2 - 1;
  median += low;

  int l = low;
  int m = high - 1;

  while (l < m) {
    KDTreeNodeInfo<DATA, DIM> elt = (*initialEltList)[median];
    int i = l;
    int j = m;

    do {
      // The even depth is associated to dim1 dimension
      // The odd one to dim2 dimension
      const unsigned thedim = treeDepth % DIM;
      while ((*initialEltList)[i].dims[thedim] < elt.dims[thedim])
        ++i;
      while ((*initialEltList)[j].dims[thedim] > elt.dims[thedim])
        --j;

      if (i <= j) {
        std::swap((*initialEltList)[i], (*initialEltList)[j]);
        i++;
        j--;
      }
    } while (i <= j);
    if (j < median)
      l = i;
    if (i > median)
      m = j;
  }

  return median;
}

template <typename DATA, unsigned int DIM>
void KDTreeLinkerAlgo<DATA, DIM>::search(const KDTreeBox<DIM> &trackBox, std::vector<DATA> &recHits) {
  if (!empty()) {
    closestNeighbour = &recHits;
    recSearch(0, trackBox, 0);
    closestNeighbour = nullptr;
  }
}

template <typename DATA, unsigned int DIM>
void KDTreeLinkerAlgo<DATA, DIM>::recSearch(int current, const KDTreeBox<DIM> &trackBox, int depth) const {
  // Iterate until leaf is found, or there are no children in the
  // search window. If search has to proceed on both children, proceed
  // the search to left child via recursion. Swap search window
  // dimension on alternate levels.
  while (true) {
    const int dimIndex = depth % DIM;
    int right = nodePool_.right[current];
    if (nodePool_.isLeaf(right)) {
      // If point inside the rectangle/area
      // Use intentionally bit-wise & instead of logical && for better
      // performance. It is faster to always do all comparisons than to
      // allow use of branches to not do some if any of the first ones
      // is false.
      bool isInside = true;
      for (unsigned i = 0; i < DIM; ++i) {
        float dimCurr = nodePool_.dims[i][current];
        isInside &= (dimCurr >= trackBox.dimmin[i]) && (dimCurr <= trackBox.dimmax[i]);
      }
      if (isInside) {
        closestNeighbour->push_back(nodePool_.data[current]);
      }
      break;
    } else {
      float median = nodePool_.dims[dimIndex][current];

      bool goLeft = (trackBox.dimmin[dimIndex] <= median);
      bool goRight = (trackBox.dimmax[dimIndex] >= median);

      ++depth;
      if (goLeft & goRight) {
        int left = current + 1;
        recSearch(left, trackBox, depth);
        // continue with right
        current = right;
      } else if (goLeft) {
        ++current;
      } else if (goRight) {
        current = right;
      } else {
        break;
      }
    }
  }
}

template <typename DATA, unsigned int DIM>
int KDTreeLinkerAlgo<DATA, DIM>::recBuild(int low, int high, int depth) {
  int portionSize = high - low;

  if (portionSize == 1) {  // Leaf case
    int leaf = nodePool_.getNextNode();
    const KDTreeNodeInfo<DATA, DIM> &info = (*initialEltList)[low];
    nodePool_.right[leaf] = 0;
    for (unsigned i = 0; i < DIM; ++i) {
      nodePool_.dims[i][leaf] = info.dims[i];
    }
    nodePool_.data[leaf] = info.data;
    return leaf;

  } else {  // Node case

    // The even depth is associated to dim1 dimension
    // The odd one to dim2 dimension
    int medianId = medianSearch(low, high, depth);
    int dimIndex = depth % DIM;
    float medianVal = (*initialEltList)[medianId].dims[dimIndex];

    // We create the node
    int nodeInd = nodePool_.getNextNode();

    ++depth;
    ++medianId;

    // We recursively build the son nodes
    int left = recBuild(low, medianId, depth);
    assert(nodeInd + 1 == left);
    int right = recBuild(medianId, high, depth);
    nodePool_.right[nodeInd] = right;

    nodePool_.dims[dimIndex][nodeInd] = medianVal;

    return nodeInd;
  }
}

#endif
