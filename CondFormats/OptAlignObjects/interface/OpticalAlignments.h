#ifndef OpticalAlignments_H
#define OpticalAlignments_H

#include "CondFormats/OptAlignObjects/interface/OpticalAlignInfo.h"

#include <vector>
#include <iostream>

/**
  easy output...
**/

class OpticalAlignments;

std::ostream & operator<<(std::ostream &, const OpticalAlignments &);

/**
   Description: Class for OpticalAlignments for use by COCOA.
 **/
class OpticalAlignments {
public:
  OpticalAlignments() {}
  virtual ~OpticalAlignments() {}

  std::vector<OpticalAlignInfo> opticalAlignments() const { return  opticalAlignments_; }

 public:
  std::vector<OpticalAlignInfo> opticalAlignments_;
};

/* typedef std::vector<int>  OptAlignIDs; */
/* typedef std::vector<int>::const_iterator OptAlignIDIterator; */

#endif // OpticalAlignments_H
