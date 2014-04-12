#ifndef CSCChannelMapperESProducer_H
#define CSCChannelMapperESProducer_H

#include <memory>
#include "boost/shared_ptr.hpp"

#include "FWCore/Framework/interface/ESProducer.h"

#include "CalibMuon/CSCCalibration/interface/CSCChannelMapperBase.h"
#include "CalibMuon/CSCCalibration/interface/CSCChannelMapperRecord.h"

class CSCChannelMapperESProducer : public edm::ESProducer {

 public:
  typedef boost::shared_ptr<CSCChannelMapperBase> BSP_TYPE;

  CSCChannelMapperESProducer(const edm::ParameterSet&);
  ~CSCChannelMapperESProducer();

  BSP_TYPE produce(const CSCChannelMapperRecord&);

 private:
  std::string algoName;
};

#endif
