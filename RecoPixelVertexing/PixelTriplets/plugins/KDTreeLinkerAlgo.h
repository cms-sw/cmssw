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
	      std::vector<KDTreeNodeInfo<DATA> >	&resRecHitList);

  // This reurns true if the tree is empty
  bool empty() {return nodePoolPos_ == -1;}

  // This returns the number of nodes + leaves in the tree
  // (nElements should be (size() +1)/2)
  int size() { return nodePoolPos_ + 1;}

  // This method clears all allocated structures.
  void clear();
  
 private:
  // The KDTree root
  KDTreeNode<DATA>*	root_;
  
  // The node pool allow us to do just 1 call to new for each tree building.
  KDTreeNode<DATA>*	nodePool_;
  int		nodePoolSize_;
  int		nodePoolPos_;


  
  std::vector<KDTreeNodeInfo<DATA> >	*closestNeighbour;
  std::vector<KDTreeNodeInfo<DATA> >	*initialEltList;
  
 private:
 
  // Get next node from the node pool.
  KDTreeNode<DATA>* getNextNode();

  //Fast median search with Wirth algorithm in eltList between low and high indexes.
  int medianSearch(int					low,
		   int					high,
		   int					treeDepth);

  // Recursif kdtree builder. Is called by build()
  KDTreeNode<DATA> *recBuild(int				low,
		       int				hight,
		       int				depth,
		       const KDTreeBox			&region);

  // Recursif kdtree search. Is called by search()
  void recSearch(const KDTreeNode<DATA>			*current,
		 const KDTreeBox			&trackBox);    

  // Add all elements of an subtree to the closest elements. Used during the recSearch().
  void addSubtree(const KDTreeNode<DATA>			*current);

  // This method frees the KDTree.     
  void clearTree();
};


//Implementation

template < typename DATA >
void
KDTreeLinkerAlgo<DATA>::build(std::vector<KDTreeNodeInfo<DATA> >	&eltList, 
			      const KDTreeBox				&region)
{
  if (eltList.size()) {
    initialEltList = &eltList;
    
    size_t size = initialEltList->size();
    nodePoolSize_ = size * 2 - 1;
    nodePool_ = new KDTreeNode<DATA>[nodePoolSize_];
    
    // Here we build the KDTree
    root_ = recBuild(0, size, 0, region);
    
    initialEltList = 0;
  }
}
 
//Fast median search with Wirth algorithm in eltList between low and high indexes.
template < typename DATA >
int
KDTreeLinkerAlgo<DATA>::medianSearch(int	low,
				     int	high,
				     int	treeDepth)
{
  //We should have at least 1 element to calculate the median...
  //assert(low < high);

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
	while ((*initialEltList)[i].dim2 < elt.dim2) i++;
	while ((*initialEltList)[j].dim2 > elt.dim2) j--;
      } else {
	while ((*initialEltList)[i].dim1 < elt.dim1) i++;
	while ((*initialEltList)[j].dim1 > elt.dim1) j--;
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
			 std::vector<KDTreeNodeInfo<DATA> > &recHits)
{
  if (root_) {
    closestNeighbour = &recHits;
    recSearch(root_, trackBox);
    closestNeighbour = 0;
  }
}


template < typename DATA >
void 
KDTreeLinkerAlgo<DATA>::recSearch(const KDTreeNode<DATA>	*current,
				  const KDTreeBox		&trackBox)
{
  /*
  // By construction, current can't be null
  assert(current != 0);

  // By Construction, a node can't have just 1 son.
  assert (!(((current->left == 0) && (current->right != 0)) ||
	    ((current->left != 0) && (current->right == 0))));
  */
    
  if ((current->left == 0) && (current->right == 0)) {//leaf case
  
    // If point inside the rectangle/area
    if ((current->info.dim1 >= trackBox.dim1min) && (current->info.dim1 <= trackBox.dim1max) &&
	(current->info.dim2 >= trackBox.dim2min) && (current->info.dim2 <= trackBox.dim2max))
      closestNeighbour->push_back(current->info);

  } else {

    //if region( v->left ) is fully contained in the rectangle
    if ((current->left->region.dim1min >= trackBox.dim1min) && 
	(current->left->region.dim1max <= trackBox.dim1max) &&
	(current->left->region.dim2min >= trackBox.dim2min) && 
	(current->left->region.dim2max <= trackBox.dim2max))
      addSubtree(current->left);
    
    else { //if region( v->left ) intersects the rectangle
      
      if (!((current->left->region.dim1min >= trackBox.dim1max) || 
	    (current->left->region.dim1max <= trackBox.dim1min) ||
	    (current->left->region.dim2min >= trackBox.dim2max) || 
	    (current->left->region.dim2max <= trackBox.dim2min)))
	recSearch(current->left, trackBox);
    }
    
    //if region( v->right ) is fully contained in the rectangle
    if ((current->right->region.dim1min >= trackBox.dim1min) && 
	(current->right->region.dim1max <= trackBox.dim1max) &&
	(current->right->region.dim2min >= trackBox.dim2min) && 
	(current->right->region.dim2max <= trackBox.dim2max))
      addSubtree(current->right);

    else { //if region( v->right ) intersects the rectangle
     
      if (!((current->right->region.dim1min >= trackBox.dim1max) || 
	    (current->right->region.dim1max <= trackBox.dim1min) ||
	    (current->right->region.dim2min >= trackBox.dim2max) || 
	    (current->right->region.dim2max <= trackBox.dim2min)))
	recSearch(current->right, trackBox);
    } 
  }
}

template < typename DATA >
void
KDTreeLinkerAlgo<DATA>::addSubtree(const KDTreeNode<DATA>	*current)
{
  // By construction, current can't be null
  // assert(current != 0);

  if ((current->left == 0) && (current->right == 0)) // leaf
    closestNeighbour->push_back(current->info);
  else { // node
    addSubtree(current->left);
    addSubtree(current->right);
  }
}




template <typename DATA>
KDTreeLinkerAlgo<DATA>::KDTreeLinkerAlgo()
  : root_ (0),
    nodePool_(0),
    nodePoolSize_(-1),
    nodePoolPos_(-1)
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
  delete[] nodePool_;
  nodePool_ = 0;
  root_ = 0;
  nodePoolSize_ = -1;
  nodePoolPos_ = -1;
}

template <typename DATA>
void 
KDTreeLinkerAlgo<DATA>::clear()
{
  if (root_)
    clearTree();
}


template <typename DATA>
KDTreeNode<DATA>* 
KDTreeLinkerAlgo<DATA>::getNextNode()
{
  ++nodePoolPos_;

  // The tree size is exactly 2 * nbrElts - 1 and this is the total allocated memory.
  // If we have used more than that....there is a big problem.
  // assert(nodePoolPos_ < nodePoolSize_);

  return &(nodePool_[nodePoolPos_]);
}


template <typename DATA>
KDTreeNode<DATA>*
KDTreeLinkerAlgo<DATA>::recBuild(int					low, 
			   int					high, 
			   int					depth,
			   const KDTreeBox&			region)
{
  int portionSize = high - low;

  // By construction, portionSize > 0 can't happend.
  // assert(portionSize > 0);

  if (portionSize == 1) { // Leaf case
   
    KDTreeNode<DATA> *leaf = getNextNode();
    leaf->setAttributs(region, (*initialEltList)[low]);
    return leaf;

  } else { // Node case
    
    // The even depth is associated to dim1 dimension
    // The odd one to dim2 dimension
    int medianId = medianSearch(low, high, depth);

    // We create the node
    KDTreeNode<DATA> *node = getNextNode();
    node->setAttributs(region);


    // Here we split into 2 halfplanes the current plane
    KDTreeBox leftRegion = region;
    KDTreeBox rightRegion = region;
    if (depth & 1) {

      auto medianVal = (*initialEltList)[medianId].dim2;
      leftRegion.dim2max = medianVal;
      rightRegion.dim2min = medianVal;

    } else {

      auto medianVal = (*initialEltList)[medianId].dim1;
      leftRegion.dim1max = medianVal;
      rightRegion.dim1min = medianVal;

    }

    ++depth;
    ++medianId;

    // We recursively build the son nodes
    node->left = recBuild(low, medianId, depth, leftRegion);
    node->right = recBuild(medianId, high, depth, rightRegion);

    return node;
  }
}

#endif
