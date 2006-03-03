#ifndef OpticalAlignInfo_H
#define OpticalAlignInfo_H

#include <string>
#include <vector>
#include <iostream>

#include "CondFormats/OptAlignObjects/interface/OAQuality.h"

/**
  easy output...
**/

class OpticalAlignInfo;

std::ostream & operator<<(std::ostream &, const OpticalAlignInfo &);

/**
  easy output...
**/

class OpticalAlignParam;

std::ostream & operator<<(std::ostream &, const OpticalAlignParam &);


/** a Class holding data for each parameter, the value, error and whether
    it is an unknown, calibrated or fixed parameter.

    Author:  Michael Case
    Date:    Dec. 19, 2005
 **/
class OpticalAlignParam {

 public:
  double value_;
  double error_;
  int qual_; // f = fixed, c = calibrated, u = unknown.
  std::string name_;
  void clear() {
    value_ = 0.0;
    error_ = 0.0;
    qual_ = int( oa_unknown );
    name_.clear();
  }
};

// a Class holding data for an Optical Alignment transformation
/**
    Author:  Michael Case
    Date:    Dec. 15, 2005

    It is my understanding that each optical geometrical object
    has a position in space and possible other parameters such as
    
 **/
class  OpticalAlignInfo {
 public:  
  OpticalAlignParam x_, y_, z_, angx_, angy_, angz_;
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
/* class  OpticalAlignCOPSInfo : public OpticalAlignInfo { */
/*  public:   */
/*   OpticalAlignParam dowel1X_, dowel1Y_; */
/*   OpticalAlignParam upCCDtoDowel2X_, upCCDtoDowel2Y_; */
/*   OpticalAlignParam downCCDtoDowel2X_, downCCDtoDowel2Y_; */
/*   OpticalAlignParam leftCCDtoDowel2X_, leftCCDtoDowel2Y_; */
/*   OpticalAlignParam rightCCDtoDowel2X_, rightCCDtoDowel2Y_; */
/* }; */

#endif //OpticalAlignInfo_H
