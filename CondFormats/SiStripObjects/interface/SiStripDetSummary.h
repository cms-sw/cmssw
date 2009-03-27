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
 * Note: consider the possibility to move this class inside SiStripBaseObject as a protected member class.
 *
 */

class SiStripDetSummary
{
public:
  void add(const DetId & detid, const float & value);
  void print(stringstream& ss) const;

protected:
  // Maps to store the LorentzAngle and the counts
  map<int, double> meanMap_;
  map<int, double> rmsMap_;
  map<int, int> countMap_;
};

#endif
