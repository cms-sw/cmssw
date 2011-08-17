#include "RecoParticleFlow/PFProducer/interface/KDTreeTrackEcalLinker.h"

using namespace KDTreeLinker;

KDTree::KDTree()
  : root_ (0)
{
}

KDTree::~KDTree()
{
  clear();
}

void
KDTree::build(std::vector<RHinfo>	&eltList, 
	      const TBox		&region)
{
  if (eltList.size()) {
    root_ = recBuild(eltList, 0, eltList.size(), 0, region);
  }
}


TNode*
KDTree::recBuild(std::vector<RHinfo>	&eltList, 
		 int			low, 
		 int			high, 
		 int			depth,
		 const TBox&		region)
{
  int portionSize = high - low;

  // TODO YG : write a clean error msg.

  //Should never happend
  if (portionSize == 0) {
    ////////////////////////////////////////////////////////////////////////////////
    std::cout << "------------------------- IMPOSSIBLE BECAME POSSIBLE" << std::endl;
    return 0;
  }

  if (portionSize == 1) { //leaf case
   
    TNode *leaf = new TNode(region, eltList[low]);
    return leaf;

  } else {//split into two halfplanes
    
    // The even depth is associated to eta dimension
    // The odd one to phi dimension
    int medianId = medianSearch(eltList, low, high, depth);

    TNode *node = new TNode(region);

    TBox leftRegion = region;
    TBox rightRegion = region;

    if (depth & 1) {

      double medianVal = eltList[medianId].phi;
      leftRegion.phimax = medianVal;
      rightRegion.phimin = medianVal;

    } else {

      double medianVal = eltList[medianId].eta;
      leftRegion.etamax = medianVal;
      rightRegion.etamin = medianVal;

    }

    ++depth;
    ++medianId;

    node->left = recBuild(eltList, low, medianId, depth, leftRegion);
    node->right = recBuild(eltList, medianId, high, depth, rightRegion);

    return node;
  }
}



//Fast median search with Wirth algorithm in eltList between low and high indexes.
int
KDTree::medianSearch(std::vector<RHinfo>	&eltList,
		     int			low,
		     int			high,
		     int			treeDepth)
{
  int nbrElts = high - low;
  int median = (nbrElts & 1)	? nbrElts / 2 
				: nbrElts / 2 - 1;
  median += low;

  int l = low;
  int m = high - 1;
  
  while (l < m) {
    RHinfo elt = eltList[median];
    int i = l;
    int j = m;

    do {

      // The even depth is associated to eta dimension
      // The odd one to phi dimension
      if (treeDepth & 1) {
	while (eltList[i].phi < elt.phi) i++;
	while (eltList[j].phi > elt.phi) j--;
      } else {
	while (eltList[i].eta < elt.eta) i++;
	while (eltList[j].eta > elt.eta) j--;
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
KDTree::swap(RHinfo &e1, 
	     RHinfo &e2)
{
  RHinfo tmp = e1;
  e1 = e2;
  e2 = tmp;
}

void
KDTree::search(const TBox		&trackBox,
	       std::vector<RHinfo>	&recHits)
{
  if (getRoot())
    recSearch(getRoot(), trackBox, recHits);
}





void 
KDTree::recSearch(const TNode		*current,
		  const TBox		&trackBox,
		  std::vector<RHinfo>	&recHits)
{
  // TODO YG : write a clean error msg.
  
  ////////////////////////////////
  //Should never happend
  if (current == 0) {
    std::cout << "------------------------- IMPOSSIBLE BECAME POSSIBLE" << std::endl;

    return;
  }
  //left == null xor right == null => Should never happend
  if (((current->left == 0) && (current->right != 0)) ||
      ((current->left != 0) && (current->right == 0)))
    std::cout << "------------------------- IMPOSSIBLE BECAME POSSIBLE" << std::endl;
  //////////////////////////////


    
  if ((current->left == 0) && (current->right == 0)) {//leaf case
  
    //if point inside the rectangle/area
    if ((current->rh.eta >= trackBox.etamin) && (current->rh.eta <= trackBox.etamax) &&
	(current->rh.phi >= trackBox.phimin) && (current->rh.phi <= trackBox.phimax))
      recHits.push_back(current->rh);

  } else {

    //if region( v->left ) is fully contained in the rectangle
    if ((current->left->region.etamin >= trackBox.etamin) && 
	(current->left->region.etamax <= trackBox.etamax) &&
	(current->left->region.phimin >= trackBox.phimin) && 
	(current->left->region.phimax <= trackBox.phimax))
      addSubtree(current->left, recHits);
    
    else { //if region( v->left ) intersects the rectangle
      
      if (!((current->left->region.etamin >= trackBox.etamax) || 
	    (current->left->region.etamax <= trackBox.etamin) ||
	    (current->left->region.phimin >= trackBox.phimax) || 
	    (current->left->region.phimax <= trackBox.phimin)))
	recSearch(current->left, trackBox, recHits);
    }
    
    //if region( v->right ) is fully contained in the rectangle
    if ((current->right->region.etamin >= trackBox.etamin) && 
	(current->right->region.etamax <= trackBox.etamax) &&
	(current->right->region.phimin >= trackBox.phimin) && 
	(current->right->region.phimax <= trackBox.phimax))
      addSubtree(current->right, recHits);

    else { //if region( v->right ) intersects the rectangle
     
      if (!((current->right->region.etamin >= trackBox.etamax) || 
	    (current->right->region.etamax <= trackBox.etamin) ||
	    (current->right->region.phimin >= trackBox.phimax) || 
	    (current->right->region.phimax <= trackBox.phimin)))
	recSearch(current->right, trackBox, recHits);
    } 
  }
}

void
KDTree::addSubtree(const TNode		*current, 
		   std::vector<RHinfo>	&recHits)
{
  // TODO YG : write a clean error msg.

  //////////////////////
  //Should never happend
  if (current == 0) 
    {    
      std::cout << "------------------------- IMPOSSIBLE BECAME POSSIBLE - addSubtree" << std::endl;
      return;
    }
  //left =null xor right =null => a tester
  ////////////////////////



  if ((current->left == 0) && (current->right == 0)) // leaf
    recHits.push_back(current->rh);
  else { // node
    addSubtree(current->left, recHits);
    addSubtree(current->right, recHits);
  }
}


void 
KDTree::clearTree(TNode *&current)
{
  if (current->left)
    clearTree(current->left);
  if (current->right)
    clearTree(current->right);
    
  delete current;
  current = 0;
}

TNode*
KDTree::getRoot()
{
  return root_;
}

void 
KDTree::clear()
{
  if (root_)
    clearTree(root_);
}


