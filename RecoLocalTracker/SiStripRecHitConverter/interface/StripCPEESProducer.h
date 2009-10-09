#ifndef RecoLocaltracker_SiStriprecHitConverter_StripCPEESProducer_h
#define RecoLocaltracker_SiStriprecHitConverter_StripCPEESProducer_h

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoLocalTracker/Records/interface/TkStripCPERecord.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"
#include <boost/shared_ptr.hpp>

class  StripCPEESProducer: public edm::ESProducer{
 public:
  StripCPEESProducer(const edm::ParameterSet & p);
  virtual ~StripCPEESProducer(); 
  boost::shared_ptr<StripClusterParameterEstimator> produce(const TkStripCPERecord &);
 private:
  boost::shared_ptr<StripClusterParameterEstimator> _cpe;
  edm::ParameterSet pset_;
};


#endif




