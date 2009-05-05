#ifndef RecoLocaltracker_SiStriprecHitConverter_StripCPEfromTrackAngleESProducer_h
#define RecoLocaltracker_SiStriprecHitConverter_StripCPEfromTrackAngleESProducer_h

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoLocalTracker/Records/interface/TkStripCPERecord.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"
#include <boost/shared_ptr.hpp>

class  StripCPEfromTrackAngleESProducer: public edm::ESProducer {
 public:
  StripCPEfromTrackAngleESProducer(const edm::ParameterSet&);
  boost::shared_ptr<StripClusterParameterEstimator> produce(const TkStripCPERecord&);
 private:
  boost::shared_ptr<StripClusterParameterEstimator> cpe_;
  edm::ParameterSet pset_;
};
#endif




