#ifndef FASTSIMULATION_TRACKING_SEEDFINDER_H
#define FASTSIMULATION_TRACKING_SEEDFINDER_H

#include "FastSimulation/Tracking/interface/SeedingTree.h"

#include <functional>

class SeedFinder
{

    public:
        typedef std::function<bool(const std::vector<const TrajectorySeedHitCandidate*>& hits)> Selector;
    private:
        Selector _selector;
        
        const SeedingTree<TrackingLayer>& _seedingTree;
    public:
        SeedFinder(const SeedingTree<TrackingLayer>& seedingTree):
            _seedingTree(seedingTree)
        {
            _selector=[](const std::vector<const TrajectorySeedHitCandidate*>& hits) -> bool
            {
                return true;
            };
        }

        void setHitSelector(Selector selector)
        {
            _selector = selector;
        }
        
        std::vector<unsigned int> getSeed(const std::vector<TrajectorySeedHitCandidate>& trackerRecHits) const
        {
            std::vector<int> hitIndicesInTree(_seedingTree.numberOfNodes(),-1);
            //A SeedingNode is associated by its index to this list. The list stores the indices of the hits in 'trackerRecHits'
	        /* example
	           SeedingNode                     | hit index                 | hit
	           -------------------------------------------------------------------------------
	           index=  0:  [BPix1]             | hitIndicesInTree[0] (=1)  | trackerRecHits[1]
	           index=  1:   -- [BPix2]         | hitIndicesInTree[1] (=3)  | trackerRecHits[3]
	           index=  2:   --  -- [BPix3]     | hitIndicesInTree[2] (=4)  | trackerRecHits[4]
	           index=  3:   --  -- [FPix1_pos] | hitIndicesInTree[3] (=6)  | trackerRecHits[6]
	           index=  4:   --  -- [FPix1_neg] | hitIndicesInTree[4] (=7)  | trackerRecHits[7]
	     
	           The implementation has been chosen such that the tree only needs to be build once upon construction.
	        */
            return iterateHits(0,trackerRecHits,hitIndicesInTree,true);
            
            //TODO: create pairs of TrackingLayer -> remove TrajectorySeedHitCandidate class
        }
        
        
        const SeedingNode<TrackingLayer>* insertHit(
            const std::vector<TrajectorySeedHitCandidate>& trackerRecHits,
            std::vector<int>& hitIndicesInTree,
            const SeedingNode<TrackingLayer>* node, unsigned int trackerHit
        ) const
        {
            if (!node->getParent() || hitIndicesInTree[node->getParent()->getIndex()]>=0)
            {
                if (hitIndicesInTree[node->getIndex()]<0)
                {
                    const TrajectorySeedHitCandidate& currentTrackerHit = trackerRecHits[trackerHit];
                    if (currentTrackerHit.getTrackingLayer()!=node->getData())
                    {
                        return nullptr;
                    }
                    
                    
                    //fill vector of Hits from node to root to be passed to the selector function
                    std::vector<const TrajectorySeedHitCandidate*> seedCandidateHitList(node->getDepth()+1);
                    seedCandidateHitList[node->getDepth()]=&currentTrackerHit;
                    const SeedingNode<TrackingLayer>* parentNode = node->getParent();
                    while (parentNode!=nullptr)
                    {
                        seedCandidateHitList[parentNode->getDepth()]=&trackerRecHits[hitIndicesInTree[parentNode->getIndex()]];
                        parentNode = parentNode->getParent();
                    }
                    
                    if (!_selector(seedCandidateHitList))
                    {
                        return nullptr;
                    }
                    
                    
                    hitIndicesInTree[node->getIndex()]=trackerHit;
                    if (node->getChildrenSize()==0)
                    {
                        return node;
                    }
                    
                    return nullptr;
                }   
                else
                {
                    for (unsigned int ichild = 0; ichild<node->getChildrenSize(); ++ichild)
                    {
                        const SeedingNode<TrackingLayer>* seed = insertHit(trackerRecHits,hitIndicesInTree,node->getChild(ichild),trackerHit);
                        if (seed)
                        {
                            return seed;
                        }
                    }
                }
            }
            return nullptr;
        }

        std::vector<unsigned int> iterateHits(
            unsigned int start,
            const std::vector<TrajectorySeedHitCandidate>& trackerRecHits,
            std::vector<int> hitIndicesInTree,
            bool processSkippedHits
        ) const
        {
            for (unsigned int irecHit = start; irecHit<trackerRecHits.size(); ++irecHit)
            {
                unsigned int currentHitIndex=irecHit;

                for (unsigned int inext=currentHitIndex+1; inext< trackerRecHits.size(); ++inext)
                {
                    //if multiple hits are on the same layer -> follow all possibilities by recusion
                    if (trackerRecHits[currentHitIndex].getTrackingLayer()==trackerRecHits[inext].getTrackingLayer())
                    {
                        if (processSkippedHits)
                        {
                            //recusively call the method again with hit 'inext' but skip all following on the same layer though 'processSkippedHits=false'
                            std::vector<unsigned int> seedHits = iterateHits(
                                inext,
                                trackerRecHits,
                                hitIndicesInTree,
                                false
                            );
                            if (seedHits.size()>0)
                            {
                                return seedHits;
                            }
                        }
                        irecHit+=1; 
                    }
                    else
                    {
                        break;
                    }
                }

                //processSkippedHits=true

                const SeedingNode<TrackingLayer>* seedNode = nullptr;
                for (unsigned int iroot=0; seedNode==nullptr && iroot<_seedingTree.numberOfRoots(); ++iroot)
                {
                    seedNode=insertHit(trackerRecHits,hitIndicesInTree,_seedingTree.getRoot(iroot), currentHitIndex);
                }
                if (seedNode)
                {
                    std::vector<unsigned int> seedIndices(seedNode->getDepth()+1);
                    while (seedNode)
                    {
                        seedIndices[seedNode->getDepth()]=hitIndicesInTree[seedNode->getIndex()];
                        seedNode=seedNode->getParent();
                    }
                    return seedIndices;
                }

            }

            return std::vector<unsigned int>();
        }
};

#endif

