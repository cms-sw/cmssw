#ifndef RecoLocaltracker_SiStriprecHitConverter_StripCPEESProducer_h
#define RecoLocaltracker_SiStriprecHitConverter_StripCPEESProducer_h

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoLocalTracker/Records/interface/TkStripCPERecord.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"
#include <boost/shared_ptr.hpp>
#include <map>
#include "FWCore/Framework/interface/DependentRecordImplementation.h"
#include "CalibTracker/Records/interface/SiStripDependentRecords.h"
#include "CalibTracker/SiStripESProducers/plugins/real/SiStripLorentzAngleDepESProducer.h"

class  StripCPEESProducer: public edm::ESProducer {

 public:

  StripCPEESProducer(const edm::ParameterSet&);
  boost::shared_ptr<StripClusterParameterEstimator> produce(const TkStripCPERecord&);

 private:

  enum CPE_t { SIMPLE, TRACKANGLE, GEOMETRIC, TEMPLATE};
  std::map<std::string,CPE_t> enumMap; 

  CPE_t cpeNum;
  edm::ParameterSet parametersPSet;
  boost::shared_ptr<StripClusterParameterEstimator> cpe;

};
#endif




