#ifndef RecoLocaltracker_SiPixelRecHits_PixelCPEInitialESProducer_h
#define RecoLocaltracker_SiPixelRecHits_PixelCPEInitialESProducer_h

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoLocalTracker/Records/interface/TkPixelCPERecord.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"
#include <boost/shared_ptr.hpp>

class  PixelCPEInitialESProducer: public edm::ESProducer{
 public:
  PixelCPEInitialESProducer(const edm::ParameterSet & p);
  virtual ~PixelCPEInitialESProducer(); 
  boost::shared_ptr<PixelClusterParameterEstimator> produce(const TkPixelCPERecord &);
 private:
  boost::shared_ptr<PixelClusterParameterEstimator> cpe_;
  edm::ParameterSet pset_;
};


#endif




