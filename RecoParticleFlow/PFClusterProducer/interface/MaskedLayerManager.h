#ifndef __MaskedLayerManager_H__
#define __MaskedLayerManager_H__

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"

#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"

#include <unordered_map>
#include <iostream>

class MaskedLayerManager {
 public:
  MaskedLayerManager(const edm::ParameterSet& conf);
  MaskedLayerManager(const MaskedLayerManager&) = delete;
  MaskedLayerManager& operator=(const MaskedLayerManager&) = delete;

  bool isRecHitDropped(const reco::PFRecHit&) const;
  std::multimap<unsigned,unsigned> buildAbsorberGanging(const ForwardSubdetector& ) const;
  const std::map<unsigned,bool> layerMask(const ForwardSubdetector& det ) const {
    if( !allowed_layers.count(det) ) {
      throw cms::Exception("UnmaskableDetector") 
        << "No mask available for " << det << std::endl;
    }
    return allowed_layers.find((int)det)->second;
  }
  
 private:
  std::unordered_map<int, std::map<unsigned,bool> > allowed_layers;
};

#endif
