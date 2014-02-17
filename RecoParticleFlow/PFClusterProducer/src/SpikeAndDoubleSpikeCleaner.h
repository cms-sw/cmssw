#ifndef __SpikeAndDoubleSpikeCleaner_H__
#define __SpikeAndDoubleSpikeCleaner_H__

#include "RecoParticleFlow/PFClusterProducer/interface/RecHitCleanerBase.h"

#include <unordered_map>

class SpikeAndDoubleSpikeCleaner : public RecHitCleanerBase {
 public:
  
  struct spike_cleaning {
    double _singleSpikeThresh;
    double _minS4S1_a;
    double _minS4S1_b;
    double _doubleSpikeS6S2;
    double _eneThreshMod;
    double _fracThreshMod;
    double _doubleSpikeThresh;
  };

  SpikeAndDoubleSpikeCleaner(const edm::ParameterSet& conf);
  SpikeAndDoubleSpikeCleaner(const SpikeAndDoubleSpikeCleaner&) = delete;
  SpikeAndDoubleSpikeCleaner& operator=(const SpikeAndDoubleSpikeCleaner&) = delete;

  void clean( const edm::Handle<reco::PFRecHitCollection>& input,
	      std::vector<bool>& mask );

 private:
  const std::unordered_map<std::string,int> _layerMap;
  std::unordered_map<int,spike_cleaning> _thresholds;
  
};

#include "RecoParticleFlow/PFClusterProducer/interface/RecHitCleanerFactory.h"
DEFINE_EDM_PLUGIN(RecHitCleanerFactory,
		  SpikeAndDoubleSpikeCleaner,"SpikeAndDoubleSpikeCleaner");

#endif
