// Author: Felice Pantaleo - felice.pantaleo@cern.ch
// Date: 09/2018
// Copyright CERN

#ifndef RecoHGCal_TICL_PatternRecognitionAlgoBase_H__
#define RecoHGCal_TICL_PatternRecognitionAlgoBase_H__

#include <vector>
#include <memory>
#include "RecoHGCal/TICL/interface/Trackster.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"

namespace edm {class ParameterSet; class Event; class EventSetup;}


class PatternRecognitionAlgoBase {
public:
  PatternRecognitionAlgoBase(const edm::ParameterSet& conf){}
  virtual ~PatternRecognitionAlgoBase() {};

  virtual void makeTracksters(
      const edm::Event& ev,
      const edm::EventSetup& es,
      const std::vector<reco::CaloCluster>& layerClusters,
      const std::vector<std::pair<unsigned int, float> >& mask, std::vector<Trackster>& result) const {};
};


#endif
