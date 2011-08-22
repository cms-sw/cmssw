#ifndef KDTreeLinkerAlgo_h
#define KDTreeLinkerAlgo_h

#include "RecoParticleFlow/PFProducer/interface/KDTreeLinkerTools.h"

#include <vector>

namespace KDTreeLinker
{
  // Class that implements the KDTree partition of space and the search algorithme.
  class KDTreeLinkerAlgo
  {
  public:
    KDTreeLinkerAlgo();

    // For now, Dtor just calls clear()
    ~KDTreeLinkerAlgo();

    void build(std::vector<RHinfo>	&eltList,
	       const TBox		&region);
    
    void search(const TBox		&trackBox,
		std::vector<RHinfo>	&recHits);

    // This method clears all allocated structures.
    void clear();
    
  private:
    // The KDTree root
    TNode*	root_;

    TNode*	nodePool_;
    int		nodePoolSize_;
    int		nodePoolPos_;

  private:
    void swap(RHinfo &e1, RHinfo &e2);

    TNode* getNextNode();


    //Fast median search with Wirth algorithm in eltList between low and high indexes.
    int medianSearch(std::vector<RHinfo>	&eltList,
		     int			low,
		     int			high,
		     int			treeDepth);

    // Recursif kdtree builder. Is called by build()
    TNode *recBuild(std::vector<RHinfo> &eltList, 
		    int			low,
		    int			hight,
		    int			depth,
		    const TBox&		region);

    // Recursif kdtree search. Is called by search()
    void recSearch(const TNode		*current,
		   const TBox		&trackBox,
		   std::vector<RHinfo>	&recHits);    

    // Add all elements of an subtree to the closest elements. Used during the recSearch().
    void addSubtree(const TNode		*current, 
		    std::vector<RHinfo> &recHits);

    // This method frees the KDTree.     
    void clearTree();
  };
}

#endif
