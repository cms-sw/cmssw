#ifndef FastSimulation_TrackingRecHitProducer_FastPixelCPEESProducer_h
#define FastSimulation_TrackingRecHitProducer_FastPixelCPEESProducer_h

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoLocalTracker/Records/interface/TkPixelCPERecord.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"
#include "FastPixelCPE.h"
#include <boost/shared_ptr.hpp>

class  FastPixelCPEESProducer: public edm::ESProducer{
 public:
  FastPixelCPEESProducer(const edm::ParameterSet & p);
  virtual ~FastPixelCPEESProducer(); 
  boost::shared_ptr<PixelClusterParameterEstimator> produce(const TkPixelCPERecord &);
 private:
  boost::shared_ptr<PixelClusterParameterEstimator> cpe_;
  edm::ParameterSet pset_;
};


#endif




