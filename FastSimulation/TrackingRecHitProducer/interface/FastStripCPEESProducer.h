#ifndef FastSimulation_TrackingRecHitProducer_FastStripCPEESProducer_h
#define FastSimulation_TrackingRecHitProducer_FastStripCPEESProducer_h

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoLocalTracker/Records/interface/TkStripCPERecord.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"
#include <boost/shared_ptr.hpp>

class  FastStripCPEESProducer: public edm::ESProducer{
 public:
  FastStripCPEESProducer(const edm::ParameterSet & p);
  ~FastStripCPEESProducer();
  boost::shared_ptr<StripClusterParameterEstimator> produce(const TkStripCPERecord &);
 private:
  boost::shared_ptr<StripClusterParameterEstimator> _cpe;
  edm::ParameterSet pset_;
};
 
 
#endif
