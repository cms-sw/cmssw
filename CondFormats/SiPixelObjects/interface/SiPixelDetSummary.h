#ifndef SiPixelDetSummary_h
#define SiPixelDetSummary_h

#include "DataFormats/DetId/interface/DetId.h"

#include <sstream>
#include <map>
#include <cmath>
#include <iomanip>
#include <iostream>

/**
 * @class SiPixelDetSummary
 * @author Urs Langenegger, using SiStripDetSummary
 * @date 2010/05/04
 * Class to compute and print pixel detector summary information
 *
 * If values are passed together with DetIds (method add( detId, value)), it computes the mean value and
 * rms of a given quantity and is able to print a summary divided by layer/disk for each subdetector. <br>
 * If instead only DetIds are passed (method add( detId )), it prints the count divided by layer/disk for
 * each subdetector. <br>
 * <br>
 */

class SiPixelDetSummary {
public:
  SiPixelDetSummary(int verbose = 0);
  
  void add(const DetId &detid, const float &value);
  void add(const DetId &detid);
  
  void print(std::stringstream &ss, const bool mean = true) const;
  
  std::map<int, int> getCounts() { return fCountMap;  }
  
protected:
  std::map<int, double> fMeanMap;
  std::map<int, double> fRmsMap;
  std::map<int, int> fCountMap;
  bool fComputeMean;
  int  fVerbose;
};

#endif
