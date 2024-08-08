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
#include "DataFormats/HGCalReco/interface/Common.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"
#include "CommonTools/RecoAlgos/interface/MultiVectorManager.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"

namespace edm {
  class Event;
  class EventSetup;
}  // namespace edm

namespace cms {
  namespace Ort {
    class ONNXRuntime;
  }
}  // namespace cms

namespace ticl {
  class TracksterLinkingAlgoBase {
  public:
    /** \param conf the configuration of the plugin
     * \param onnxRuntime the ONNXRuntime, if onnxModelPath was provided in plugin configuration (nullptr otherwise)
    */
    TracksterLinkingAlgoBase(const edm::ParameterSet& conf,
                             edm::ConsumesCollector,
                             cms::Ort::ONNXRuntime const* onnxRuntime = nullptr)
        : algo_verbosity_(conf.getParameter<int>("algo_verbosity")), onnxRuntime_(onnxRuntime) {}
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
             const MultiVectorManager<Trackster>& tS)
          : ev(eV), es(eS), layerClusters(lC), layerClustersTime(lT), tracksters(tS) {}
    };

    virtual void linkTracksters(const Inputs& input,
                                std::vector<Trackster>& resultTracksters,
                                std::vector<std::vector<unsigned int>>& linkedResultTracksters,
                                std::vector<std::vector<unsigned int>>& linkedTracksterIdToInputTracksterId) = 0;

    virtual void initialize(const HGCalDDDConstants* hgcons,
                            const hgcal::RecHitTools rhtools,
                            const edm::ESHandle<MagneticField> bfieldH,
                            const edm::ESHandle<Propagator> propH) = 0;

    // To be called by TracksterLinksProducer at the start of TracksterLinksProducer::produce. Subclasses can use this to store Event and EventSetup
    virtual void setEvent(edm::Event& iEvent, edm::EventSetup const& iEventSetup){};

    static void fillPSetDescription(edm::ParameterSetDescription& desc) { desc.add<int>("algo_verbosity", 0); };

  protected:
    int algo_verbosity_;
    cms::Ort::ONNXRuntime const* onnxRuntime_;
  };
}  // namespace ticl

#endif
