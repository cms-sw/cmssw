#ifndef RecoLocaltracker_SiStriprecHitConverter_StripCPEfromTrackAngleESProducer_h
#define RecoLocaltracker_SiStriprecHitConverter_StripCPEfromTrackAngleESProducer_h

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoLocalTracker/Records/interface/TrackerCPERecord.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"
#include <boost/shared_ptr.hpp>

class  StripCPEfromTrackAngleESProducer: public edm::ESProducer{
 public:
  StripCPEfromTrackAngleESProducer(const edm::ParameterSet & p);
  virtual ~StripCPEfromTrackAngleESProducer(); 
  boost::shared_ptr<StripClusterParameterEstimator> produce(const TrackerCPERecord &);
 private:
  boost::shared_ptr<StripClusterParameterEstimator> _cpe;
  edm::ParameterSet pset_;
};


#endif




