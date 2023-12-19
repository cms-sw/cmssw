#ifndef RecoHGCal_TICL_PatternRecognitionAlgoBase_H__
#define RecoHGCal_TICL_PatternRecognitionAlgoBase_H__

#include <memory>
#include <vector>
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/HGCalReco/interface/TICLCandidate.h"
#include "DataFormats/HGCalReco/interface/TICLLayerTile.h"
#include "DataFormats/HGCalReco/interface/TICLSeedingRegion.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "RecoHGCal/TICL/interface/GlobalCache.h"
#include "RecoHGCal/TICL/interface/commons.h"
#include "FWCore/Framework/interface/ESConsumesCollector.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"
#include "CommonTools/RecoAlgos/interface/MultiVectorManager.h"

namespace edm {
  class Event;
  class EventSetup;
}  // namespace edm
namespace ticl {
  template<typename T>
  class TICLInterpretationAlgoBase {
  public:
    TICLInterpretationAlgoBase(const edm::ParameterSet& conf, edm::ConsumesCollector, edm::ESConsumesCollector)
        : algo_verbosity_(conf.getParameter<int>("algo_verbosity")) {}
    virtual ~TICLInterpretationAlgoBase(){};
    struct Inputs {
      const edm::Event& ev;
      const edm::EventSetup& es;
      const std::vector<reco::CaloCluster>& layerClusters;
      const edm::ValueMap<std::pair<float, float>>& layerClustersTime;
      const MultiVectorManager<Trackster>& tracksters;
      const std::vector<std::vector<unsigned int>> & linkedResultTracksters;
      const std::vector<T>& tracks;
      
      Inputs(const edm::Event& eV,
             const edm::EventSetup& eS,
             const std::vector<reco::CaloCluster>& lC,
             const edm::ValueMap<std::pair<float, float>>& lT,
             const MultiVectorManager<Trackster>& tS,
             const std::vector<std::vector<unsigned int>> & links,
             const std::vector<T>& trks )
          : ev(eV), es(eS), layerClusters(lC), layerClustersTime(lT), tracksters(tS), linkedResultTracksters(links), tracks(trks) {}
    };

    virtual void makeCandidates(const Inputs& input, std::vector<Trackster>& resultTracksters,
                                std::vector<TICLCandidate>& resultCandidate) = 0;

    static void fillPSetDescription(edm::ParameterSetDescription& desc) { desc.add<int>("algo_verbosity", 0); };

  protected:
    int algo_verbosity_;
  };
}  // namespace ticl

#endif
