#ifndef OpticalAlignInfo_H
#define OpticalAlignInfo_H

#include "CondFormats/Serialization/interface/Serializable.h"

#include <string>
#include <vector>
#include <iostream>

#include "CondFormats/OptAlignObjects/interface/OAQuality.h"

/**
  easy output...
**/

class OpticalAlignInfo;

std::ostream &operator<<(std::ostream &, const OpticalAlignInfo &);

/**
   easy output...
**/

class OpticalAlignParam;

std::ostream &operator<<(std::ostream &, const OpticalAlignParam &);

/** a Class holding data for each parameter, the value, error and whether
    it is an unknown, calibrated or fixed parameter.

    Author:  Michael Case
    Date:    Dec. 19, 2005
 **/
class OpticalAlignParam {
public:
  OpticalAlignParam();

  std::string name() const { return name_; }
  double value() const { return value_; }
  double sigma() const { return error_; }
  int quality() const { return quality_; }
  std::string dimType() const { return dim_type_; }

public:
  double value_;
  double error_;
  int quality_;  // f = fixed, c = calibrated, u = unknown.
  std::string name_;
  std::string dim_type_;

  void clear() {
    value_ = 0.0;
    error_ = 0.0;
    quality_ = int(oa_unknown);
    name_.clear();
  }

  COND_SERIALIZABLE;
};

// a Class holding data for an Optical Alignment transformation
/**
    Author:  Michael Case
    Date:    Dec. 15, 2005

    It is my understanding that each optical geometrical object
    has a position in space and possible other parameters such as
    
 **/
class OpticalAlignInfo {
public:
  /*
  OpticalAlignParam x() const { return x_; }
  OpticalAlignParam y() const { return y_; }
  OpticalAlignParam z() const { return z_; }
  OpticalAlignParam angX() const { return angx_; }
  OpticalAlignParam angY() const { return angy_; }
  OpticalAlignParam angZ() const { return angz_; }
  std::vector<OpticalAlignParam> extraEntries() const { return extraEntries_; }
  std::string type() { return type_; }
  std::string name() const { return name_; }
  std::string parentName() const { return parentObjectName_; }
  unsigned int ID() const { return ID_; }
  */
  OpticalAlignParam *findExtraEntry(std::string &name);

public:
  OpticalAlignParam x_, y_, z_, angx_, angy_, angz_;
  std::vector<OpticalAlignParam> extraEntries_;
  std::string type_;
  std::string name_;
  std::string parentName_;
  unsigned int ID_;
  void clear() {
    x_.clear();
    y_.clear();
    z_.clear();
    angx_.clear();
    angy_.clear();
    angz_.clear();
    extraEntries_.clear();
    type_.clear();
    ID_ = 0;
  }

  COND_SERIALIZABLE;
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

#endif  //OpticalAlignInfo_H
