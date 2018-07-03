#ifndef KDTreeLinkerAlgoTemplated_h
#define KDTreeLinkerAlgoTemplated_h

#include "KDTreeLinkerTools.h"

#include <cassert>
#include <vector>

// Class that implements the KDTree partition of 2D space and 
// a closest point search algorithme.

template <typename DATA>
class KDTreeLinkerAlgo
{
 public:
  KDTreeLinkerAlgo();
  
  // Dtor calls clear()
  ~KDTreeLinkerAlgo();
  
  // Here we build the KD tree from the "eltList" in the space define by "region".
  void build(std::vector<KDTreeNodeInfo<DATA> > 	&eltList,
	     const KDTreeBox				&region);
  
  // Here we search in the KDTree for all points that would be 
  // contained in the given searchbox. The founded points are stored in resRecHitList.
  void search(const KDTreeBox				&searchBox,
	      std::vector<DATA>	&resRecHitList);

  // This reurns true if the tree is empty
  bool empty() {return nodePool_.empty();}

  // This returns the number of nodes + leaves in the tree
  // (nElements should be (size() +1)/2)
  int size() { return nodePool_.size();}

  // This method clears all allocated structures.
  void clear();
  
 private:
  // The node pool allow us to do just 1 call to new for each tree building.
  KDTreeNodes<DATA> nodePool_;
  
  std::vector<DATA>	*closestNeighbour;
  std::vector<KDTreeNodeInfo<DATA> >	*initialEltList;
  
 private:
 
  //Fast median search with Wirth algorithm in eltList between low and high indexes.
  int medianSearch(int					low,
		   int					high,
		   int					treeDepth);

  // Recursif kdtree builder. Is called by build()
  int recBuild(int				low,
               int				hight,
               int				depth);

  // Recursif kdtree search. Is called by search()
  void recSearch(int			current,
                 float dimCurrMin, float dimCurrMax,
                 float dimOtherMin, float dimOtherMax);

  // This method frees the KDTree.     
  void clearTree();
};


//Implementation

template < typename DATA >
void
KDTreeLinkerAlgo<DATA>::build(std::vector<KDTreeNodeInfo<DATA> >	&eltList, 
			      const KDTreeBox				&region)
{
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
template < typename DATA >
int
KDTreeLinkerAlgo<DATA>::medianSearch(int	low,
				     int	high,
				     int	treeDepth)
{
  int nbrElts = high - low;
  int median = (nbrElts & 1)	? nbrElts / 2 
				: nbrElts / 2 - 1;
  median += low;

  int l = low;
  int m = high - 1;
  
  while (l < m) {
    KDTreeNodeInfo<DATA> elt = (*initialEltList)[median];
    int i = l;
    int j = m;

    do {
      // The even depth is associated to dim1 dimension
      // The odd one to dim2 dimension
      if (treeDepth & 1) {
	while ((*initialEltList)[i].dim[1] < elt.dim[1]) i++;
	while ((*initialEltList)[j].dim[1] > elt.dim[1]) j--;
      } else {
	while ((*initialEltList)[i].dim[0] < elt.dim[0]) i++;
	while ((*initialEltList)[j].dim[0] > elt.dim[0]) j--;
      }

      if (i <= j){
	std::swap((*initialEltList)[i], (*initialEltList)[j]);
	i++; 
	j--;
      }
    } while (i <= j);
    if (j < median) l = i;
    if (i > median) m = j;
  }

  return median;
}



template < typename DATA >
void
KDTreeLinkerAlgo<DATA>::search(const KDTreeBox		&trackBox,
			 std::vector<DATA> &recHits)
{
  if (!empty()) {
    closestNeighbour = &recHits;
    recSearch(0, trackBox.dim1min, trackBox.dim1max, trackBox.dim2min, trackBox.dim2max);
    closestNeighbour = nullptr;
  }
}


template < typename DATA >
void 
KDTreeLinkerAlgo<DATA>::recSearch(int	current,
                                  float dimCurrMin, float dimCurrMax,
                                  float dimOtherMin, float dimOtherMax)
{
  // Iterate until leaf is found, or there are no children in the
  // search window. If search has to proceed on both children, proceed
  // the search to left child via recursion. Swap search window
  // dimension on alternate levels.
  while(true) {
    int right = nodePool_.right[current];
    if(nodePool_.isLeaf(right)) {
      float dimCurr = nodePool_.median[current];

      // If point inside the rectangle/area
      // Use intentionally bit-wise & instead of logical && for better
      // performance. It is faster to always do all comparisons than to
      // allow use of branches to not do some if any of the first ones
      // is false.
      if((dimCurr >= dimCurrMin) & (dimCurr <= dimCurrMax)) {
        float dimOther = nodePool_.dimOther[current];
        if((dimOther >= dimOtherMin) & (dimOther <= dimOtherMax)) {
          closestNeighbour->push_back(nodePool_.data[current]);
        }
      }
      break;
    }
    else {
      float median = nodePool_.median[current];

      bool goLeft = (dimCurrMin <= median);
      bool goRight = (dimCurrMax >= median);

      // Swap dimension for the next search level
      std::swap(dimCurrMin, dimOtherMin);
      std::swap(dimCurrMax, dimOtherMax);
      if(goLeft & goRight) {
        int left = current+1;
        recSearch(left, dimCurrMin, dimCurrMax, dimOtherMin, dimOtherMax);
        // continue with right
        current = right;
      }
      else if(goLeft) {
        ++current;
      }
      else if(goRight) {
        current = right;
      }
      else {
        break;
      }
    }
  }
}

template <typename DATA>
KDTreeLinkerAlgo<DATA>::KDTreeLinkerAlgo()
{
}

template <typename DATA>
KDTreeLinkerAlgo<DATA>::~KDTreeLinkerAlgo()
{
  clear();
}


template <typename DATA>
void 
KDTreeLinkerAlgo<DATA>::clearTree()
{
  nodePool_.clear();
}

template <typename DATA>
void 
KDTreeLinkerAlgo<DATA>::clear()
{
  clearTree();
}


template <typename DATA>
int
KDTreeLinkerAlgo<DATA>::recBuild(int					low, 
                                 int					high, 
                                 int					depth)
{
  int portionSize = high - low;
  int dimIndex = depth&1;

  if (portionSize == 1) { // Leaf case
    int leaf = nodePool_.getNextNode();
    const KDTreeNodeInfo<DATA>& info = (*initialEltList)[low];
    nodePool_.right[leaf] = 0;
    nodePool_.median[leaf] = info.dim[dimIndex]; // dimCurrent
    nodePool_.dimOther[leaf] = info.dim[1-dimIndex];
    nodePool_.data[leaf] = info.data;
    return leaf;

  } else { // Node case
    
    // The even depth is associated to dim1 dimension
    // The odd one to dim2 dimension
    int medianId = medianSearch(low, high, depth);
    float medianVal = (*initialEltList)[medianId].dim[dimIndex];

    // We create the node
    int nodeInd = nodePool_.getNextNode();
    nodePool_.median[nodeInd] = medianVal;

    ++depth;
    ++medianId;

    // We recursively build the son nodes
    int left = recBuild(low, medianId, depth);
    assert(nodeInd+1 == left);
    nodePool_.right[nodeInd] = recBuild(medianId, high, depth);

    return nodeInd;
  }
}

#endif
