#ifndef RecoLocaltracker_SiStriprecHitConverter_StripCPEESProducer_h
#define RecoLocaltracker_SiStriprecHitConverter_StripCPEESProducer_h

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoLocalTracker/Records/interface/TkStripCPERecord.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"
#include <boost/shared_ptr.hpp>
#include <map>

class  StripCPEESProducer: public edm::ESProducer {

 public:

  StripCPEESProducer(const edm::ParameterSet&);
  boost::shared_ptr<StripClusterParameterEstimator> produce(const TkStripCPERecord&);

 private:

  enum CPE_t { SIMPLE, TRACKANGLE, GEOMETRIC};
  std::map<std::string,CPE_t> enumMap; 

  CPE_t cpeNum;
  edm::ParameterSet pset;
  boost::shared_ptr<StripClusterParameterEstimator> cpe;

};
#endif




