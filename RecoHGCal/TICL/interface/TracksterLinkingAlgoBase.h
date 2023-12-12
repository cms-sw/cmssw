// Author: Felice Pantaleo - felice.pantaleo@cern.ch
// Date: 09/2018

#ifndef RecoHGCal_TICL_PatternRecognitionAlgoBase_H__
#define RecoHGCal_TICL_PatternRecognitionAlgoBase_H__

#include <memory>
#include <vector>
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/HGCalReco/interface/TICLLayerTile.h"
#include "DataFormats/HGCalReco/interface/TICLSeedingRegion.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "RecoHGCal/TICL/interface/GlobalCache.h"
#include "RecoHGCal/TICL/interface/commons.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"
#include "CommonTools/RecoAlgos/interface/MultiVectorManager.h"

namespace edm {
  class Event;
  class EventSetup;
}  // namespace edm

namespace ticl {
  class TracksterLinkingAlgoBase {
  public:
    TracksterLinkingAlgoBase(const edm::ParameterSet& conf, edm::ConsumesCollector)
        : algo_verbosity_(conf.getParameter<int>("algo_verbosity")) {}
    virtual ~TracksterLinkingAlgoBase(){};

    struct Inputs {
      const edm::Event& ev;
      const edm::EventSetup& es;
      const std::vector<reco::CaloCluster>& layerClusters;
      const edm::ValueMap<std::pair<float, float>>& layerClustersTime;
      const MultiVectorManager<Trackster>& tracksters;
      
      Inputs(const edm::Event& eV,
             const edm::EventSetup& eS,
             const std::vector<reco::CaloCluster>& lC,
             const edm::ValueMap<std::pair<float, float>>& lT,
             const MultiVectorManager<Trackster>& tS )
          : ev(eV), es(eS), layerClusters(lC), layerClustersTime(lT), tracksters(tS) {}
    };

    virtual void linkTracksters(const Inputs& input, std::vector<Trackster>& resultTracksters,
                                std::vector<std::vector<unsigned int>> & linkedResultTracksters,
                                std::vector<std::vector<unsigned int>>& linkedTracksterIdToInputTracksterId) = 0;

    static void fillPSetDescription(edm::ParameterSetDescription& desc) { desc.add<int>("algo_verbosity", 0); };

  protected:
    int algo_verbosity_;
  };
}  // namespace ticl

#endif
