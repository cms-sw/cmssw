#ifndef SiPixelCommon_SiPixelHistogramId_h
#define SiPixelCommon_SiPixelHistogramId_h
// -*- C++ -*-
//
// Package:     SiPixelCommon
// Class  :     SiPixelHistogramId
//
/**\class SiPixelHistogramId SiPixelHistogramId.h
 DQM/SiPixelCommon/interface/SiPixelHistogramId.h

 Description: Creates and returns DQM Histogram Id's

 Usage:
    <usage>

*/
//
// Original Author:  chiochia
//         Created:  Wed Feb 22 16:07:51 CET 2006
//

#include <string>
#include <cstdint>

class SiPixelHistogramId {
public:
  /// Constructor
  SiPixelHistogramId();
  /// Constructor
  SiPixelHistogramId(std::string dataCollection);
  /// Destructor
  virtual ~SiPixelHistogramId();
  /// Set Histogram Id
  std::string setHistoId(std::string variable, uint32_t &rawId);
  /// Get data Collection
  std::string getDataCollection(std::string histogramId);
  /// Get Detector Raw Id
  uint32_t getRawId(std::string histogramId);

private:
  std::string returnIdPart(std::string histoid, uint32_t whichpart);
  std::string dataCollection_;
  std::string separator_;
};

#endif
