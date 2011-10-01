#ifndef KDTreeLinkerAlgo_h
#define KDTreeLinkerAlgo_h

#include "RecoParticleFlow/PFProducer/interface/KDTreeLinkerTools.h"

#include <vector>

// Class that implements the KDTree partition of 2D space and 
// a closest point search algorithme.
class KDTreeLinkerAlgo
{
 public:
  KDTreeLinkerAlgo();
  
  // Dtor calls clear()
  ~KDTreeLinkerAlgo();
  
  // Here we build the KD tree from the "eltList" in the space define by "region".
  void build(std::vector<KDTreeNodeInfo>	&eltList,
	     const KDTreeBox			&region);
  
  // Here we search in the KDTree for all points that would be 
  // contained in the given searchbox. The founded points are stored in resRecHitList.
  void search(const KDTreeBox			&searchBox,
	      std::vector<KDTreeNodeInfo>	&resRecHitList);
  
  // This method clears all allocated structures.
  void clear();
  
 private:
  // The KDTree root
  KDTreeNode*	root_;
  
  // The node pool allow us to do just 1 call to new for each tree building.
  KDTreeNode*	nodePool_;
  int		nodePoolSize_;
  int		nodePoolPos_;

 private:
  // Basic swap function.
  void swap(KDTreeNodeInfo &e1, KDTreeNodeInfo &e2);

  // Get next node from the node pool.
  KDTreeNode* getNextNode();

  //Fast median search with Wirth algorithm in eltList between low and high indexes.
  int medianSearch(std::vector<KDTreeNodeInfo>	&eltList,
		   int				low,
		   int				high,
		   int				treeDepth);

  // Recursif kdtree builder. Is called by build()
  KDTreeNode *recBuild(std::vector<KDTreeNodeInfo>	&eltList, 
		       int				low,
		       int				hight,
		       int				depth,
		       const KDTreeBox&			region);

  // Recursif kdtree search. Is called by search()
  void recSearch(const KDTreeNode		*current,
		 const KDTreeBox		&trackBox,
		 std::vector<KDTreeNodeInfo>	&recHits);    

  // Add all elements of an subtree to the closest elements. Used during the recSearch().
  void addSubtree(const KDTreeNode		*current, 
		  std::vector<KDTreeNodeInfo>	&recHits);

  // This method frees the KDTree.     
  void clearTree();
};

#endif
