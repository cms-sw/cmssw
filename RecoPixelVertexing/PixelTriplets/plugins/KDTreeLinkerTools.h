#ifndef KDTreeLinkerToolsTemplated_h
#define KDTreeLinkerToolsTemplated_h

#include <assert.h>
#include <algorithm>

// Box structure used to define 2D field.
// It's used in KDTree building step to divide the detector
// space (ECAL, HCAL...) and in searching step to create a bounding
// box around the demanded point (Track collision point, PS projection...).
struct KDTreeBox
{
  float dim1min, dim1max;
  float dim2min, dim2max;
  
  public:

  KDTreeBox(float d1min, float d1max, 
	    float d2min, float d2max)
    : dim1min (d1min), dim1max(d1max)
    , dim2min (d2min), dim2max(d2max)
  {}

  KDTreeBox()
    : dim1min (0), dim1max(0)
    , dim2min (0), dim2max(0)
  {}
};

  
// Data stored in each KDTree node.
// The dim1/dim2 fields are usually the duplication of some PFRecHit values
// (eta/phi or x/y). But in some situations, phi field is shifted by +-2.Pi
template <typename DATA>
struct KDTreeNodeInfo 
{
  DATA data;
  float dim[2];
  enum {kDim1=0, kDim2=1};

  public:
  KDTreeNodeInfo()
  {}
  
  KDTreeNodeInfo(const DATA&	d,
		 float		dim_1,
		 float		dim_2)
    : data(d), dim{dim_1, dim_2}
  {}
};

// KDTree node.
template <typename DATA>
struct KDTreeNode
{
  // Data
  KDTreeNodeInfo<DATA> info;
  
  // Right/left sons.
  KDTreeNode *left, *right;
  
  // Region bounding box.
  KDTreeBox region;
  
  public:
  KDTreeNode()
    : left(0), right(0)
  {}
  
  void setAttributs(const KDTreeBox&		regionBox,
		    const KDTreeNodeInfo<DATA>&	infoToStore) 
  {
    info = infoToStore;
    region = regionBox;
  }
  
  void setAttributs(const KDTreeBox&   regionBox) 
  {
    region = regionBox;
  }
};

template <typename DATA>
struct KDTreeNodes {
  struct Node {
    Node(): median(0.0f), right(0) {}
    float median;
    int right;
  };
  std::vector<Node> nodes;
  std::vector<KDTreeNodeInfo<DATA>> info;

  int poolSize;
  int poolPos;

  constexpr KDTreeNodes(): poolSize(-1), poolPos(-1) {}

  bool empty() const { return poolPos == -1; }
  int size() const { return poolPos + 1; }

  void clear() {
    nodes.clear();
    info.clear();
    poolSize = -1;
    poolPos = -1;
  }

  int getNextNode() {
    ++poolPos;
    // The tree size is exactly 2 * nbrElts - 1 and this is the total allocated memory.
    // If we have used more than that....there is a big problem.
    //assert(poolPos < poolSize);
    return poolPos;
  }

  void build(int sizeData) {
    poolSize = sizeData*2-1;
    nodes.resize(poolSize);
    info.resize(poolSize);
  };

  bool isLeaf(int index) const {
    // Valid values of right are always >= 2
    // index 0 is the root, and 1 is the first left node
    // Exploit index values 0 and 1 to mark which of dim1/dim2 is the
    // current one in recSearch() at the depth of the leaf.
    return nodes[index].right < 2;
  }
};

#endif
