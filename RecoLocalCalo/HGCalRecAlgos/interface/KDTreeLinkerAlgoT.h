#ifndef KDTreeLinkerAlgoTemplated_h
#define KDTreeLinkerAlgoTemplated_h

#include "KDTreeLinkerToolsT.h"

#include <cassert>
#include <vector>

// Class that implements the KDTree partition of 2D space and 
// a closest point search algorithme.

template <typename DATA, unsigned DIM=2>
class KDTreeLinkerAlgo
{
 public:
  KDTreeLinkerAlgo();
  
  // Dtor calls clear()
  ~KDTreeLinkerAlgo();
  
  // Here we build the KD tree from the "eltList" in the space define by "region".
 void build(std::vector<KDTreeNodeInfoT<DATA,DIM> > 	&eltList,
	    const KDTreeBoxT<DIM>	                &region);
  
  // Here we search in the KDTree for all points that would be 
  // contained in the given searchbox. The founded points are stored in resRecHitList.
  void search(const KDTreeBoxT<DIM>			&searchBox,
	      std::vector<KDTreeNodeInfoT<DATA,DIM> >	&resRecHitList);

  // This reurns true if the tree is empty
  bool empty() {return nodePoolPos_ == -1;}

  // This returns the number of nodes + leaves in the tree
  // (nElements should be (size() +1)/2)
  int size() { return nodePoolPos_ + 1;}

  // This method clears all allocated structures.
  void clear();
  
 private:
 // The KDTree root
 KDTreeNodeT<DATA,DIM>*	root_;
 
 // The node pool allow us to do just 1 call to new for each tree building.
 KDTreeNodeT<DATA,DIM>*	nodePool_;
 int		nodePoolSize_;
 int		nodePoolPos_;


  
 std::vector<KDTreeNodeInfoT<DATA,DIM> >	*closestNeighbour;
 std::vector<KDTreeNodeInfoT<DATA,DIM> >	*initialEltList;
  
 private:
 
  // Get next node from the node pool.
 KDTreeNodeT<DATA,DIM>* getNextNode();

  //Fast median search with Wirth algorithm in eltList between low and high indexes.
  int medianSearch(int					low,
		   int					high,
		   int					treeDepth);

  // Recursif kdtree builder. Is called by build()
 KDTreeNodeT<DATA,DIM> *recBuild(int				low,
				 int				hight,
				 int				depth,
				 const KDTreeBoxT<DIM>		&region);

  // Recursif kdtree search. Is called by search()
 void recSearch(const KDTreeNodeT<DATA,DIM>		*current,
		const KDTreeBoxT<DIM>			&trackBox);    

  // Add all elements of an subtree to the closest elements. Used during the recSearch().
 void addSubtree(const KDTreeNodeT<DATA,DIM>		*current);
 
 // This method frees the KDTree.     
 void clearTree();
};


//Implementation

template < typename DATA, unsigned DIM >
void
KDTreeLinkerAlgo<DATA,DIM>::build(std::vector<KDTreeNodeInfoT<DATA,DIM> >  &eltList, 
				  const KDTreeBoxT<DIM>  		  &region)
{
  if (eltList.size()) {
    initialEltList = &eltList;
    
    size_t size = initialEltList->size();
    nodePoolSize_ = size * 2 - 1;
    nodePool_ = new KDTreeNodeT<DATA,DIM>[nodePoolSize_];
    
    // Here we build the KDTree
    root_ = recBuild(0, size, 0, region);
    
    initialEltList = 0;
  }
}
 
//Fast median search with Wirth algorithm in eltList between low and high indexes.
template < typename DATA, unsigned DIM >
int
KDTreeLinkerAlgo<DATA,DIM>::medianSearch(int	low,
					 int	high,
					 int	treeDepth)
{
  //We should have at least 1 element to calculate the median...
  //assert(low < high);

  const int nbrElts = high - low;  
  int median = nbrElts/2 - ( 1 - 1*(nbrElts&1) );
  median += low;

  int l = low;
  int m = high - 1;
  
  while (l < m) {
    KDTreeNodeInfoT<DATA,DIM> elt = (*initialEltList)[median];
    int i = l;
    int j = m;

    do {
      // The even depth is associated to dim1 dimension
      // The odd one to dim2 dimension
      const unsigned thedim = treeDepth % DIM;
      while( (*initialEltList)[i].dims[thedim] < elt.dims[thedim] ) ++i;
      while( (*initialEltList)[j].dims[thedim] > elt.dims[thedim] ) --j;      

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



template < typename DATA, unsigned DIM >
void
KDTreeLinkerAlgo<DATA,DIM>::search(const KDTreeBoxT<DIM>		  &trackBox,
				   std::vector<KDTreeNodeInfoT<DATA,DIM> > &recHits)
{
  if (root_) {
    closestNeighbour = &recHits;
    recSearch(root_, trackBox);
    closestNeighbour = 0;
  }
}


template < typename DATA, unsigned DIM >
void 
KDTreeLinkerAlgo<DATA,DIM>::recSearch(const KDTreeNodeT<DATA,DIM> *current,
				      const KDTreeBoxT<DIM>	 &trackBox)
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
    bool isInside = true;
    for( unsigned i = 0; i < DIM; ++i ) {
      const auto thedim = current->info.dims[i];
      isInside *= thedim >= trackBox.dimmin[i] && thedim <= trackBox.dimmax[i];
    }
    if( isInside ) closestNeighbour->push_back(current->info);

  } else {

    bool isFullyContained = true;
    bool hasIntersection = true;
    //if region( v->left ) is fully contained in the rectangle
    for( unsigned i = 0; i < DIM; ++i ) {
      const auto regionmin = current->left->region.dimmin[i];
      const auto regionmax = current->left->region.dimmax[i];
      isFullyContained *= ( regionmin >= trackBox.dimmin[i] && 
			    regionmax <= trackBox.dimmax[i]    );
      hasIntersection *= ( regionmin < trackBox.dimmax[i] && regionmax > trackBox.dimmin[i]);
    }
    if( isFullyContained ) {
      addSubtree(current->left);
    } else if ( hasIntersection ) {
      recSearch(current->left, trackBox); 
    }    
    // reset flags
    isFullyContained = true;
    hasIntersection = true;
    //if region( v->right ) is fully contained in the rectangle
    for( unsigned i = 0; i < DIM; ++i ) {
      const auto regionmin = current->right->region.dimmin[i];
      const auto regionmax = current->right->region.dimmax[i];
      isFullyContained *= ( regionmin >= trackBox.dimmin[i] && 
			    regionmax <= trackBox.dimmax[i]    );
      hasIntersection *= ( regionmin < trackBox.dimmax[i] && regionmax > trackBox.dimmin[i]);
    }
    if( isFullyContained ) {
      addSubtree(current->right);
    } else if ( hasIntersection ) {
      recSearch(current->right, trackBox); 
    }    
  }
}

template < typename DATA, unsigned DIM >
void
KDTreeLinkerAlgo<DATA,DIM>::addSubtree(const KDTreeNodeT<DATA,DIM>	*current)
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




template <typename DATA, unsigned DIM>
KDTreeLinkerAlgo<DATA,DIM>::KDTreeLinkerAlgo()
  : root_ (0),
    nodePool_(0),
    nodePoolSize_(-1),
    nodePoolPos_(-1)
{
}

template <typename DATA, unsigned DIM>
KDTreeLinkerAlgo<DATA,DIM>::~KDTreeLinkerAlgo()
{
  clear();
}


template <typename DATA, unsigned DIM>
void 
KDTreeLinkerAlgo<DATA,DIM>::clearTree()
{
  delete[] nodePool_;
  nodePool_ = 0;
  root_ = 0;
  nodePoolSize_ = -1;
  nodePoolPos_ = -1;
}

template <typename DATA, unsigned DIM>
void 
KDTreeLinkerAlgo<DATA,DIM>::clear()
{
  if (root_)
    clearTree();
}


template <typename DATA, unsigned DIM>
KDTreeNodeT<DATA,DIM>* 
KDTreeLinkerAlgo<DATA,DIM>::getNextNode()
{
  ++nodePoolPos_;

  // The tree size is exactly 2 * nbrElts - 1 and this is the total allocated memory.
  // If we have used more than that....there is a big problem.
  // assert(nodePoolPos_ < nodePoolSize_);

  return &(nodePool_[nodePoolPos_]);
}


template <typename DATA, unsigned DIM>
KDTreeNodeT<DATA,DIM>*
KDTreeLinkerAlgo<DATA,DIM>::recBuild(int					low, 
				     int					high, 
				     int					depth,
				     const KDTreeBoxT<DIM>&			region)
{
  int portionSize = high - low;

  // By construction, portionSize > 0 can't happend.
  // assert(portionSize > 0);

  if (portionSize == 1) { // Leaf case
   
    KDTreeNodeT<DATA,DIM> *leaf = getNextNode();
    leaf->setAttributs(region, (*initialEltList)[low]);
    return leaf;

  } else { // Node case
    
    // The even depth is associated to dim1 dimension
    // The odd one to dim2 dimension
    int medianId = medianSearch(low, high, depth);

    // We create the node
    KDTreeNodeT<DATA,DIM> *node = getNextNode();
    node->setAttributs(region);


    // Here we split into 2 halfplanes the current plane
    KDTreeBoxT<DIM> leftRegion = region;
    KDTreeBoxT<DIM> rightRegion = region;
    unsigned thedim = depth % DIM;
    auto medianVal = (*initialEltList)[medianId].dims[thedim];
    leftRegion.dimmax[thedim] = medianVal;
    rightRegion.dimmin[thedim] = medianVal;    

    ++depth;
    ++medianId;

    // We recursively build the son nodes
    node->left = recBuild(low, medianId, depth, leftRegion);
    node->right = recBuild(medianId, high, depth, rightRegion);

    return node;
  }
}

#endif
