#ifndef CSCChannelMapperESProducer_H
#define CSCChannelMapperESProducer_H

#include <memory>

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"  // for fillDescriptions

#include "CalibMuon/CSCCalibration/interface/CSCChannelMapperBase.h"
#include "CalibMuon/CSCCalibration/interface/CSCChannelMapperRecord.h"

class CSCChannelMapperESProducer : public edm::ESProducer {
public:
  typedef std::unique_ptr<CSCChannelMapperBase> BSP_TYPE;

  CSCChannelMapperESProducer(const edm::ParameterSet &);
  ~CSCChannelMapperESProducer() override;

  BSP_TYPE produce(const CSCChannelMapperRecord &);

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  std::string algoName;
};

#endif
