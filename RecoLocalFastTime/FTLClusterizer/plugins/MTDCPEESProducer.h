#ifndef RecoLocalFastTime_FTLClusterizer_MTDCPEGenericESProducer_h
#define RecoLocalFastTime_FTLClusterizer_MTDCPEGenericESProducer_h

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoLocalFastTime/Records/interface/MTDCPERecord.h"
#include "RecoLocalFastTime/FTLClusterizer/interface/MTDClusterParameterEstimator.h"
#include <memory>

class  MTDCPEESProducer: public edm::ESProducer
{
 public:
  MTDCPEESProducer(const edm::ParameterSet & p);
  ~MTDCPEESProducer() override; 
  std::unique_ptr<MTDClusterParameterEstimator> produce(const MTDCPERecord &);
  
 private:
  edm::ParameterSet pset_;
};


#endif
