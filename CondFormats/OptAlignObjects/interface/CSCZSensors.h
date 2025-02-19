#ifndef CSCZSensors_H
#define CSCZSensors_H
/* #include "CondFormats/OptAlignObjects/interface/OpticalAlignInfo.h" */
#include <vector>
#include <string>

/**
   easy output...
**/

/* class CSCZSensors; */

/* std::ostream & operator<<(std::ostream &, const CSCZSensors &); */

class CSCZSensorData {
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
};

/**
   Description: Class for CSCZSensors for use as calibration.
**/
class CSCZSensors {
 public:
  CSCZSensors() {}
  virtual ~CSCZSensors() {}
  std::vector<CSCZSensorData> cscZSens_;
};

#endif // CSCZSensors_H
