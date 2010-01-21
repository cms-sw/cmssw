#ifndef RecoLocaltracker_SiStriprecHitConverter_StripCPEfromTrackAngle2ESProducer_h
#define RecoLocaltracker_SiStriprecHitConverter_StripCPEfromTrackAngle2ESProducer_h

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoLocalTracker/Records/interface/TkStripCPERecord.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"
#include <boost/shared_ptr.hpp>

class  StripCPEfromTrackAngle2ESProducer: public edm::ESProducer{
 public:
  StripCPEfromTrackAngle2ESProducer(const edm::ParameterSet & p);
  virtual ~StripCPEfromTrackAngle2ESProducer(); 
  boost::shared_ptr<StripClusterParameterEstimator> produce(const TkStripCPERecord &);
 private:
  boost::shared_ptr<StripClusterParameterEstimator> _cpe;
  edm::ParameterSet pset_;
};


#endif




