#ifndef CSCIndexerESProducer_H
#define CSCIndexerESProducer_H

#include <memory>
#include "boost/shared_ptr.hpp"

#include "FWCore/Framework/interface/ESProducer.h"

#include "CalibMuon/CSCCalibration/interface/CSCIndexerBase.h"
#include "CalibMuon/CSCCalibration/interface/CSCIndexerRecord.h"

class CSCIndexerESProducer : public edm::ESProducer {

 public:
  typedef boost::shared_ptr<CSCIndexerBase> BSP_TYPE;

  CSCIndexerESProducer(const edm::ParameterSet&);
  ~CSCIndexerESProducer();

  BSP_TYPE produce(const CSCIndexerRecord&);

 private:
  std::string algoName;
};

#endif
