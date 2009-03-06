#ifndef RecoBTag_PerformanceFW_BtagPerformanceESProducer_H
#define RecoBTag_PerformanceFW_BtagPerformanceESProducer_H

#include "RecoBTag/PerformanceDB/interface/BtagPerformance.h"
#include "RecoBTag/Records/interface/BTagPerformanceRecord.h"


#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <boost/shared_ptr.hpp>

class  BtagPerformanceESProducer : public edm::ESProducer{
 public:
  BtagPerformanceESProducer(const edm::ParameterSet & p);
  virtual ~BtagPerformanceESProducer(); 
  boost::shared_ptr<BtagPerformance> produce(const  BTagPerformanceRecord &);
 private:
  boost::shared_ptr<BtagPerformance> _perf;
  edm::ParameterSet pset_;
  std::string mypl;
  std::string mywp;
};


#endif




