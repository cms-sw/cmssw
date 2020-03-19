#ifndef RecoBTag_PerformanceFW_BtagPerformanceESProducer_H
#define RecoBTag_PerformanceFW_BtagPerformanceESProducer_H

#include "RecoBTag/PerformanceDB/interface/BtagPerformance.h"
#include "RecoBTag/Records/interface/BTagPerformanceRecord.h"

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <memory>

class BtagPerformanceESProducer : public edm::ESProducer {
public:
  BtagPerformanceESProducer(const edm::ParameterSet &p);
  ~BtagPerformanceESProducer() override;
  std::unique_ptr<BtagPerformance> produce(const BTagPerformanceRecord &);

private:
  edm::ParameterSet pset_;
  std::string mypl;
  std::string mywp;
};

#endif
