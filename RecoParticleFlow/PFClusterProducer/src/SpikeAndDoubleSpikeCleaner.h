#ifndef __SpikeAndDoubleSpikeCleaner_H__
#define __SpikeAndDoubleSpikeCleaner_H__

#include "RecoParticleFlow/PFClusterProducer/interface/RecHitCleanerBase.h"

class SpikeAndDoubleSpikeCleaner : public RecHitCleanerBase {
 public:
  SpikeAndDoubleSpikeCleaner(const edm::ParameterSet& conf) :
    RecHitCleanerBase(conf),
    _minS4S1_a(conf.getParameter<double>("minS4S1_a")),
    _minS4S1_b(conf.getParameter<double>("minS4S1_b")),
    _doubleSpikeS6S2(conf.getParameter<double>("doubleSpikeS6S2")),
    _eneThreshMod(conf.getParameter<double>("energyThresholdModifier")),
    _fracThreshMod(conf.getParameter<double>("fractionThresholdModifier")),
    _doubleSpikeThresh(conf.getParameter<double>("doubleSpikeThresh")) { }
  SpikeAndDoubleSpikeCleaner(const SpikeAndDoubleSpikeCleaner&) = delete;
  SpikeAndDoubleSpikeCleaner& operator=(const SpikeAndDoubleSpikeCleaner&) = delete;

  void clean( const edm::Handle<reco::PFRecHitCollection>& input,
	      std::vector<bool>& mask );

 private:
  const double _minS4S1_a;
  const double _minS4S1_b;  
  const double _doubleSpikeS6S2;
  const float _eneThreshMod;
  const float _fracThreshMod;
  const float _doubleSpikeThresh;
  
};

#include "RecoParticleFlow/PFClusterProducer/interface/RecHitCleanerFactory.h"
DEFINE_EDM_PLUGIN(RecHitCleanerFactory,
		  SpikeAndDoubleSpikeCleaner,"SpikeAndDoubleSpikeCleaner");

#endif
