#include "RecoParticleFlow/PFProducer/interface/KDTreeLinkerAlgo.h"

KDTreeLinkerAlgo::KDTreeLinkerAlgo()
  : root_ (0),
    nodePool_(0),
    nodePoolSize_(-1),
    nodePoolPos_(-1)
{
}

KDTreeLinkerAlgo::~KDTreeLinkerAlgo()
{
  clear();
}

void
KDTreeLinkerAlgo::build(std::vector<KDTreeNodeInfo>	&eltList, 
			const KDTreeBox			&region)
{
  if (eltList.size()) {
    nodePoolSize_ = eltList.size() * 2 - 1;
    nodePool_ = new KDTreeNode[nodePoolSize_];

    // Here we build the KDTree
    root_ = recBuild(eltList, 0, eltList.size(), 0, region);
  }
}


KDTreeNode*
KDTreeLinkerAlgo::recBuild(std::vector<KDTreeNodeInfo>	&eltList, 
			   int				low, 
			   int				high, 
			   int				depth,
			   const KDTreeBox&		region)
{
  int portionSize = high - low;

  // By construction, portionSize > 0 can't happend.
  assert(portionSize > 0);

  if (portionSize == 1) { // Leaf case
   
    KDTreeNode *leaf = getNextNode();
    leaf->setAttributs(region, eltList[low]);
    return leaf;

  } else { // Node case
    
    // The even depth is associated to dim1 dimension
    // The odd one to dim2 dimension
    int medianId = medianSearch(eltList, low, high, depth);

    // We create the node
    KDTreeNode *node = getNextNode();
    node->setAttributs(region);


    // Here we split into 2 halfplanes the current plane
    KDTreeBox leftRegion = region;
    KDTreeBox rightRegion = region;
    if (depth & 1) {

      double medianVal = eltList[medianId].dim2;
      leftRegion.dim2max = medianVal;
      rightRegion.dim2min = medianVal;

    } else {

      double medianVal = eltList[medianId].dim1;
      leftRegion.dim1max = medianVal;
      rightRegion.dim1min = medianVal;

    }

    ++depth;
    ++medianId;

    // We recursively build the son nodes
    node->left = recBuild(eltList, low, medianId, depth, leftRegion);
    node->right = recBuild(eltList, medianId, high, depth, rightRegion);

    return node;
  }
}


//Fast median search with Wirth algorithm in eltList between low and high indexes.
int
KDTreeLinkerAlgo::medianSearch(std::vector<KDTreeNodeInfo>	&eltList,
			       int				low,
			       int				high,
			       int				treeDepth)
{
  //We should have at least 1 element to calculate the median...
  assert(low < high);

  int nbrElts = high - low;
  int median = (nbrElts & 1)	? nbrElts / 2 
				: nbrElts / 2 - 1;
  median += low;

  int l = low;
  int m = high - 1;
  
  while (l < m) {
    KDTreeNodeInfo elt = eltList[median];
    int i = l;
    int j = m;

    do {

      // The even depth is associated to dim1 dimension
      // The odd one to dim2 dimension
      if (treeDepth & 1) {
	while (eltList[i].dim2 < elt.dim2) i++;
	while (eltList[j].dim2 > elt.dim2) j--;
      } else {
	while (eltList[i].dim1 < elt.dim1) i++;
	while (eltList[j].dim1 > elt.dim1) j--;
      }

      if (i <= j){
	swap(eltList[i], eltList[j]);
	i++; 
	j--;
      }
    } while (i <= j);
    if (j < median) l = i;
    if (i > median) m = j;
  }

  return median;
}

void 
KDTreeLinkerAlgo::swap(KDTreeNodeInfo	&e1, 
		       KDTreeNodeInfo	&e2)
{
  KDTreeNodeInfo tmp = e1;
  e1 = e2;
  e2 = tmp;
}

void
KDTreeLinkerAlgo::search(const KDTreeBox		&trackBox,
			 std::vector<KDTreeNodeInfo>	&recHits)
{
  if (root_)
    recSearch(root_, trackBox, recHits);
}

void 
KDTreeLinkerAlgo::recSearch(const KDTreeNode		*current,
			    const KDTreeBox		&trackBox,
			    std::vector<KDTreeNodeInfo>	&recHits)
{
  // By construction, current can't be null
  assert(current != 0);

  // By Construction, a node can't have just 1 son.
  assert (!(((current->left == 0) && (current->right != 0)) ||
	    ((current->left != 0) && (current->right == 0))));
    
  if ((current->left == 0) && (current->right == 0)) {//leaf case
  
    // If point inside the rectangle/area
    if ((current->rh.dim1 >= trackBox.dim1min) && (current->rh.dim1 <= trackBox.dim1max) &&
	(current->rh.dim2 >= trackBox.dim2min) && (current->rh.dim2 <= trackBox.dim2max))
      recHits.push_back(current->rh);

  } else {

    //if region( v->left ) is fully contained in the rectangle
    if ((current->left->region.dim1min >= trackBox.dim1min) && 
	(current->left->region.dim1max <= trackBox.dim1max) &&
	(current->left->region.dim2min >= trackBox.dim2min) && 
	(current->left->region.dim2max <= trackBox.dim2max))
      addSubtree(current->left, recHits);
    
    else { //if region( v->left ) intersects the rectangle
      
      if (!((current->left->region.dim1min >= trackBox.dim1max) || 
	    (current->left->region.dim1max <= trackBox.dim1min) ||
	    (current->left->region.dim2min >= trackBox.dim2max) || 
	    (current->left->region.dim2max <= trackBox.dim2min)))
	recSearch(current->left, trackBox, recHits);
    }
    
    //if region( v->right ) is fully contained in the rectangle
    if ((current->right->region.dim1min >= trackBox.dim1min) && 
	(current->right->region.dim1max <= trackBox.dim1max) &&
	(current->right->region.dim2min >= trackBox.dim2min) && 
	(current->right->region.dim2max <= trackBox.dim2max))
      addSubtree(current->right, recHits);

    else { //if region( v->right ) intersects the rectangle
     
      if (!((current->right->region.dim1min >= trackBox.dim1max) || 
	    (current->right->region.dim1max <= trackBox.dim1min) ||
	    (current->right->region.dim2min >= trackBox.dim2max) || 
	    (current->right->region.dim2max <= trackBox.dim2min)))
	recSearch(current->right, trackBox, recHits);
    } 
  }
}

void
KDTreeLinkerAlgo::addSubtree(const KDTreeNode		*current, 
		   std::vector<KDTreeNodeInfo>	&recHits)
{
  // By construction, current can't be null
  assert(current != 0);

  if ((current->left == 0) && (current->right == 0)) // leaf
    recHits.push_back(current->rh);
  else { // node
    addSubtree(current->left, recHits);
    addSubtree(current->right, recHits);
  }
}


void 
KDTreeLinkerAlgo::clearTree()
{
  delete[] nodePool_;
  nodePool_ = 0;
  root_ = 0;
  nodePoolSize_ = -1;
  nodePoolPos_ = -1;
}

void 
KDTreeLinkerAlgo::clear()
{
  if (root_)
    clearTree();
}


KDTreeNode* 
KDTreeLinkerAlgo::getNextNode()
{
  ++nodePoolPos_;

  // The tree size is exactly 2 * nbrElts - 1 and this is the total allocated memory.
  // If we have used more than that....there is a big problem.
  assert(nodePoolPos_ < nodePoolSize_);

  return &(nodePool_[nodePoolPos_]);
}
