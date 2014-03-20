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
			int _hitNumber;
			bool _active;
			std::vector<LayerNode*> _children;
			const LayerNode* _parent;
		public:
			LayerNode():
				_layer(0),
				_hitNumber(-1),
				_active(true),
				_parent(0)
			{
			}

			LayerNode(const LayerSpec* layer, const LayerNode* parent):
				_layer(layer),
				_hitNumber(-1),
				_active(true),
				_parent(parent)
			{
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
				_children.push_back(layerNode);
				return layerNode;
			}

			const LayerSpec* getLayer() const
			{
				return _layer;
			}

			bool isActive() const
			{
				return _active;
			}

			void setHitNumber(int ihit)
			{
				_hitNumber=ihit;
			}

			int getHitNumber() const
			{
				return _hitNumber;
			}

			const LayerNode* getParent() const
			{
				return _parent;
			}

			void fill(std::vector<LayerSpec>& layerSpecList)
			{
				LayerNode* currentNode = this;
				for (unsigned int i = 0; i < layerSpecList.size() && currentNode!=0; ++i)
				{
					currentNode=currentNode->addChild(&layerSpecList[i]);
				}
			}

			void reset()
			{
				_hitNumber=-1;
				_active=true;
				for (unsigned int ilayer = 0; ilayer< _children.size(); ++ilayer)
				{
					_children[ilayer]->reset();
				}
			}

			std::string str(unsigned int offset=0) const
			{
				std::stringstream ss;
				for (unsigned int i=0; i<offset;++i)
				{
					ss<<"  ";
				}
				if (_layer!=0)
				{
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

	LayerNode rootLayerNode;

 public:
  
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
  virtual bool passTrackerRecHitQualityCuts(std::vector<unsigned int> previousHits, TrackerRecHit& currentHit, unsigned int trackingAlgorithmId);

    //! method iterates over TrackerRectHits and check if those are on the requested seeding layers. Method will call itself if a track produced two hits on the same layer.
    /*!
    \param start starting position of an iterator over the hit range associated to a SimHit.
    \param range the hit range associated to a SimHit.
    \param hitNumbers list of indexes of hits which are on the seeding layers per seeding layer set.
    \param trackerRecHits list of TrackerRectHits to which the indexes in \e hitNumbers correspond to.
    \param trackingAlgorithmId id of the seeding algorithm iteration (e.g. "initial step", etc.).
    \param seedHitNumbers if a valid seed was found this list is used to store the hit indexes.
    \return the index of the layer set which produced a valid seed (same indexing used in \e hitNumbers). -1 is returned if no valid seed was found.
    */
  virtual int iterateHits(
	SiTrackerGSMatchedRecHit2DCollection::const_iterator start,
	SiTrackerGSMatchedRecHit2DCollection::range range,
	std::vector<std::vector<unsigned int>> hitNumbers,
	unsigned int trackingAlgorithmId,
	std::vector<unsigned int>& seedHitNumbers
  );

};

#endif
