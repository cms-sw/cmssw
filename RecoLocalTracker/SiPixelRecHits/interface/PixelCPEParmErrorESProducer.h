#ifndef RecoLocaltracker_SiPixelRecHits_PixelCPEParmErrorESProducer_h
#define RecoLocaltracker_SiPixelRecHits_PixelCPEParmErrorESProducer_h

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoLocalTracker/Records/interface/TrackerCPERecord.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"
#include <boost/shared_ptr.hpp>

class  PixelCPEParmErrorESProducer: public edm::ESProducer{
 public:
  PixelCPEParmErrorESProducer(const edm::ParameterSet & p);
  virtual ~PixelCPEParmErrorESProducer(); 
  boost::shared_ptr<PixelClusterParameterEstimator> produce(const TrackerCPERecord &);
 private:
  boost::shared_ptr<PixelClusterParameterEstimator> cpe_;
  edm::ParameterSet pset_;
};


#endif




