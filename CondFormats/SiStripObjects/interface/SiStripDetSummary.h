#ifndef SiStripDetSummary_h
#define SiStripDetSummary_h

#include "DataFormats/SiStripDetId/interface/TIDDetId.h" 
#include "DataFormats/SiStripDetId/interface/TECDetId.h" 
#include "DataFormats/SiStripDetId/interface/TIBDetId.h" 
#include "DataFormats/SiStripDetId/interface/TOBDetId.h" 
#include "DataFormats/DetId/interface/DetId.h"

#include <sstream>
#include <map>
#include <cmath>
#include <iomanip>
#include <iostream>

using namespace std;

/**
 * @class SiStripDetSummary
 * @author M. De Mattia
 * @date 26/3/2009
 * Class to compute and print summary information.
 *
 * If values are passed together with DetIds (method add( detId, value)), it computes the mean value and
 * rms of a given quantity and is able to print a summary divided by layer/disk for each subdetector. <br>
 * If instead only DetIds are passed (method add( detId )), it prints the count divided by layer/disk for
 * each subdetector. <br>
 * <br>
 * Note: consider the possibility to move this class inside SiStripBaseObject as a protected member class.
 *
 */

class SiStripDetSummary
{
public:
  SiStripDetSummary() : computeMean_(true) {}

  /// Used to compute the mean value of the value variable divided by subdetector, layer and mono/stereo
  void add(const DetId & detid, const float & value);
  /// Used to compute the number of entries divided by subdetector, layer and mono/stereo
  inline void add(const DetId & detid)
  {
    computeMean_ = false;
    add( detid, 0 );
  }

  /**
   * Method used to write the output. By default mean == true and it writes the mean value. If mean == false
   * it will write the count.
   */
  void print(stringstream& ss, const bool mean = true) const;

protected:
  // Maps to store the value and the counts
  map<int, double> meanMap_;
  map<int, double> rmsMap_;
  map<int, int> countMap_;
  bool computeMean_;
};

#endif
