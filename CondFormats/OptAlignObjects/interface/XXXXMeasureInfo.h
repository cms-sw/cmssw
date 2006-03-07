#ifndef XXXXMeasureInfo_H
#define XXXXMeasureInfo_H

#include <string>
#include <vector>
#include <iostream>

#include "CondFormats/OptAlignObjects/interface/OAQuality.h"
#include "CondFormats/OptAlignObjects/interface/XXXXMeasurements.h"

/**
  easy output...
**/

class XXXXMeasureInfo;

std::ostream & operator<<(std::ostream &, const XXXXMeasureInfo &);

/**
  easy output...
**/

class OpticalAlignParam;

std::ostream & operator<<(std::ostream &, const OpticalAlignParam &);


// a Class holding data for an Optical Alignment transformation
/**
    Author:  Michael Case
    Date:    Dec. 15, 2005

    It is my understanding that each optical geometrical object
    has a position in space and possible other parameters such as
    
 **/
class  XXXXMeasureInfo {
 public:  
  OpticalAlignParam x1_, x2_, x3_, x4_;
  std::vector<OpticalAlignParam> extraEntries_;
  std::string objectType_;
  unsigned long objectID_;
  void clear() {
    x_.clear();
    y_.clear();
    z_.clear();
    angx_.clear();
    angy_.clear();
    angz_.clear();
    extraEntries_.clear();
    objectType_.clear();
    objectID_ = 0;
  }
};


/**
    Author:  Michael Case
    Date:    Dec. 15, 2005

    It is my understanding that each optical geometrical object
    has a position in space and possible other parameters such as
    
 **/
/* class  OpticalAlignCOPSInfo : public XXXXMeasureInfo { */
/*  public:   */
/*   OpticalAlignParam dowel1X_, dowel1Y_; */
/*   OpticalAlignParam upCCDtoDowel2X_, upCCDtoDowel2Y_; */
/*   OpticalAlignParam downCCDtoDowel2X_, downCCDtoDowel2Y_; */
/*   OpticalAlignParam leftCCDtoDowel2X_, leftCCDtoDowel2Y_; */
/*   OpticalAlignParam rightCCDtoDowel2X_, rightCCDtoDowel2Y_; */
/* }; */

#endif //XXXXMeasureInfo_H
