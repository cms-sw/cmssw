#ifndef FastSimulation_Tracking_TrajectorySeedProducer2_h
#define FastSimulation_Tracking_TrajectorySeedProducer2_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "FastSimulation/Tracking/plugins/TrajectorySeedProducer.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "FastSimulation/Tracking/interface/TrackerRecHit.h"

#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSMatchedRecHit2DCollection.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

#include <vector>
#include <sstream>

namespace edm { 
  class ParameterSet;
  class Event;
  class EventSetup;
}

class TrajectorySeedProducer2 : public TrajectorySeedProducer
{
private:
	std::vector<TrackerRecHit> trackerRecHits;

	class LayerNode
	{
		private:
			const LayerSpec* _layer;
			int _globalIndex;
			unsigned int _depth;
			bool _active;
			std::vector<LayerNode*> _children;
			LayerNode* _parent;
			LayerNode* _nextsibling;
		public:
			LayerNode():
				_layer(0),
				_globalIndex(-1),
				_depth(0),
				_active(true),
				_parent(0),
				_nextsibling(0)
			{
			}

			LayerNode(const LayerSpec* layer, LayerNode* parent):
				_layer(layer),
				_globalIndex(-1),
				_depth(0),
				_active(true),
				_parent(parent),
				_nextsibling(0)
			{
				LayerNode* parentNode = this;
				while(parentNode->getLayer()!=0)
				{
					++_depth;
					parentNode=parentNode->getParent();
				}
			}

			int setupGlobalIndexing()
			{
				int index=0;
				LayerNode* currentNode = this->getFirstChild();
				while (currentNode!=0)
				{
					currentNode->_globalIndex=index;
					++index;
					currentNode=currentNode->next();
				}
				return index;
			}

			int getGlobalIndex()
			{
				return _globalIndex;
			}

			LayerNode* next()
			{
				if (this->getFirstChild()!=0)
				{
					return this->getFirstChild();
				}
				else
				{
					if (this->nextSibling()!=0)
					{
						return this->nextSibling();
					}
					else
					{
						LayerNode* parent = this->getParent();
						while (parent!=0)
						{
							if (parent->nextSibling()!=0)
							{
								return parent->nextSibling();
							}
							else
							{
								parent=parent->getParent();
							}
						}
						return 0;
					}
				}
			}

			LayerNode* addChild(LayerSpec* layer)
			{
				for (unsigned int ichild = 0; ichild<_children.size(); ++ichild)
				{
					if ((*layer)==(*_children[ichild]->getLayer()))
					{
						return _children[ichild];
					}
				}
				LayerNode* layerNode = new LayerNode(layer,this);
				if (_children.size()>0)
				{
					_children.back()->setSibling(layerNode);
				}
				_children.push_back(layerNode);
				return layerNode;
			}

			unsigned int getDepth() const
			{
				return _depth;
			}

			void setSibling(LayerNode* layerNode)
			{
				_nextsibling=layerNode;
			}

			const LayerSpec* getLayer() const
			{
				return _layer;
			}

			bool isActive() const
			{
				return _active;
			}

			LayerNode* getParent(unsigned int up=1)
			{
				LayerNode* node = this;
				for (unsigned int i = 0; i<up;++i)
				{
					node=node->_parent;
				}
				return node;
			}

			void fill(std::vector<LayerSpec>& layerSpecList)
			{
				LayerNode* currentNode = this;
				for (unsigned int i = 0; i < layerSpecList.size() && currentNode!=0; ++i)
				{
					currentNode=currentNode->addChild(&layerSpecList[i]);
				}
			}

			LayerNode* getFirstChild()
			{
				if (_children.size()>0)
				{
					return _children[0];
				}
				return 0;
			}

			LayerNode* nextSibling()
			{
				if (_nextsibling!=0)
				{
					return _nextsibling;
				}
				return 0;
			}

			std::string str(unsigned int offset=0) const
			{
				std::stringstream ss;
				for (unsigned int i=0; i<offset;++i)
				{
					ss<<"   ";
				}
				if (_layer!=0)
				{
					ss<<"["<<_globalIndex<<"] ";
					ss<< _layer->name;
					ss<<_layer->idLayer;

				}
				else
				{
					ss<<"0";
				}
				ss<<"\n";
				for (unsigned int ichild = 0; ichild<_children.size(); ++ichild)
				{
					ss<<_children[ichild]->str(offset+1);
				}
				return ss.str();
			}
	};

	LayerNode _rootLayerNode;
	int _maxGlobalIndex;
 public:

	virtual ~TrajectorySeedProducer2()
	{
		//std::cout<<"oldcalls: "<<oldCalls<<", newcalls: "<<newCalls<<std::endl;
	}
  
  explicit TrajectorySeedProducer2(const edm::ParameterSet& conf);
  
  virtual void produce(edm::Event& e, const edm::EventSetup& es) override;

    //! method checks if a SimTrack fulfills the requirements of the current seeding algorithm iteration.
    /*!
    \param theSimTrack the SimTrack to be tested.
    \param theSimVertex the associated SimVertex of the SimTrack.
    \param trackingAlgorithmId id of the seeding algorithm iteration (e.g. "initial step", etc.).
    \return true if a track fulfills the requirements.
    */
  virtual bool passSimTrackQualityCuts(const SimTrack& theSimTrack, const SimVertex& theSimVertex, unsigned int trackingAlgorithmId);
  
     //! method checks if a TrackerRecHit fulfills the requirements of the current seeding algorithm iteration.
    /*!
    \param trackerRecHits list of all TrackerRecHits.
    \param previousHits list of indexes of hits which already got accepted before.
    \param currentHit the current hit which needs to pass the criteria in addition to those in \e previousHits.
    \param trackingAlgorithmId id of the seeding algorithm iteration (e.g. "initial step", etc.).
    \return true if a hit fulfills the requirements.
    */

  bool passTrackerRecHitQualityCuts(LayerNode& lastnodeofseed, std::vector<int> globalHitNumbers, unsigned int trackingAlgorithmId);
  virtual int iterateHits(
	SiTrackerGSMatchedRecHit2DCollection::const_iterator start,
	SiTrackerGSMatchedRecHit2DCollection::range range,
	std::vector<int> globalHitNumbers,
	unsigned int trackingAlgorithmId,
	std::vector<unsigned int>& seedHitNumbers
  );

};

#endif
