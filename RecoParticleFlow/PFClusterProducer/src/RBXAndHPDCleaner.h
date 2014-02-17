#ifndef __RBXAndHPDCleaner_H__
#define __RBXAndHPDCleaner_H__

#include "RecoParticleFlow/PFClusterProducer/interface/RecHitCleanerBase.h"

#include <unordered_map>

class RBXAndHPDCleaner : public RecHitCleanerBase {
 public:
  RBXAndHPDCleaner(const edm::ParameterSet& conf) :
    RecHitCleanerBase(conf) { }
  RBXAndHPDCleaner(const RBXAndHPDCleaner&) = delete;
  RBXAndHPDCleaner& operator=(const RBXAndHPDCleaner&) = delete;

  void clean( const edm::Handle<reco::PFRecHitCollection>& input,
	      std::vector<bool>& mask );

 private:  
  std::unordered_map<int,std::vector<unsigned> > _hpds, _rbxs;
};

#include "RecoParticleFlow/PFClusterProducer/interface/RecHitCleanerFactory.h"
DEFINE_EDM_PLUGIN(RecHitCleanerFactory,
		  RBXAndHPDCleaner,"RBXAndHPDCleaner");

#endif
