#ifndef CondFormats_CSCRSensors_H
#define CondFormats_CSCRSensors_H

#include "CondFormats/Serialization/interface/Serializable.h"

#include <vector>
//#include <iostream>
#include <string>

class CSCRSensorData {
public:
  std::string sensorType_;
  int sensorNo_;
  std::string meLayer_;
  std::string logicalAlignmentName_;
  std::string cernDesignator_;
  std::string cernBarcode_;
  float absSlope_;
  float absSlopeError_;
  float normSlope_;
  float normSlopeError_;
  float absIntercept_;
  float absInterceptError_;
  float normIntercept_;
  float normInterceptError_;
  float shifts_;

  COND_SERIALIZABLE;
};

/**
   Description: Class for CSCRSensors for use as calibration.
 **/
class CSCRSensors {
public:
  CSCRSensors() {}
  virtual ~CSCRSensors() {}
  std::vector<CSCRSensorData> cscRSens_;

  COND_SERIALIZABLE;
};

#endif  // CondFormats_CSCRSensors_H
