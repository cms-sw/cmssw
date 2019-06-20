// Author: Felice Pantaleo - felice.pantaleo@cern.ch
// Date: 09/2018

#ifndef RecoHGCal_TICL_PatternRecognitionAlgoBase_H__
#define RecoHGCal_TICL_PatternRecognitionAlgoBase_H__

#include <memory>
#include <vector>
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/RecoHGCal/interface/Common.h"
#include "DataFormats/RecoHGCal/interface/Trackster.h"
#include "DataFormats/RecoHGCal/interface/TICLLayerTile.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edm {
  class Event;
  class EventSetup;
}  // namespace edm

namespace ticl {
  class PatternRecognitionAlgoBase {
  public:
    PatternRecognitionAlgoBase(const edm::ParameterSet& conf)
        : algo_verbosity_(conf.getParameter<int>("algo_verbosity")) {}
    virtual ~PatternRecognitionAlgoBase(){};

    virtual void makeTracksters(const edm::Event& ev,
                                const edm::EventSetup& es,
                                const std::vector<reco::CaloCluster>& layerClusters,
                                const std::vector<float>& mask,
                                const ticl::TICLLayerTiles& tiles,
                                std::vector<Trackster>& result) = 0;
    enum VerbosityLevel { None = 0, Basic, Advanced, Expert, Guru };

  protected:
    int algo_verbosity_;
  };
}  // namespace ticl

#endif
