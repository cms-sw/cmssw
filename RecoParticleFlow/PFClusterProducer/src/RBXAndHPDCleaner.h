#ifndef __RBXAndHPDCleaner_H__
#define __RBXAndHPDCleaner_H__

#include "RecoParticleFlow/PFClusterProducer/interface/RecHitTopologicalCleanerBase.h"

#include <unordered_map>

class RBXAndHPDCleaner : public RecHitTopologicalCleanerBase {
 public:
  RBXAndHPDCleaner(const edm::ParameterSet& conf) :
    RecHitTopologicalCleanerBase(conf) { }
  RBXAndHPDCleaner(const RBXAndHPDCleaner&) = delete;
  RBXAndHPDCleaner& operator=(const RBXAndHPDCleaner&) = delete;

  void clean( const edm::Handle<reco::PFRecHitCollection>& input,
	      std::vector<bool>& mask );

 private:  
  std::unordered_map<int,std::vector<unsigned> > _hpds, _rbxs;
};

DEFINE_EDM_PLUGIN(RecHitTopologicalCleanerFactory,
		  RBXAndHPDCleaner,"RBXAndHPDCleaner");

#endif
