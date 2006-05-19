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
/*   std::string name() const { return name_; } */
/*   double value() const { return value_; } */
/*   double error() const { return error_; } */
/*   int qual() const { return qual_; } */
/*   std::string dimension() const { return dimension_; } */

  OpticalAlignParam() { }
  virtual ~OpticalAlignParam() { }

  //copy constructor
/*   OpticalAlignParam( OpticalAlignParam& rhs ); */
/*   OpticalAlignParam( const OpticalAlignParam& rhs ); */

 public:
  double value_;
  double error_;
  int qual_; // f = fixed, c = calibrated, u = unknown.
  std::string name_;
  std::string dimension_;

/*   void clear() { */
/*     value_ = 0.0; */
/*     error_ = 0.0; */
/*     qual_ = int( oa_unknown ); */
/*     name_.clear(); */
/*   } */
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
  /*
  OpticalAlignParam x() const { return x_; }
  OpticalAlignParam y() const { return y_; }
  OpticalAlignParam z() const { return z_; }
  OpticalAlignParam angX() const { return angx_; }
  OpticalAlignParam angY() const { return angy_; }
  OpticalAlignParam angZ() const { return angz_; }
  std::vector<OpticalAlignParam> extraEntries() const { return extraEntries_; }
  std::string type() { return objectType_; }
  std::string name() const { return objectName_; }
  std::string parentName() const { return parentObjectName_; }
  unsigned long ID() const { return objectID_; }
  */

  OpticalAlignInfo () { } // : extraEntries_() { }
  virtual ~OpticalAlignInfo () { }

  // copy constructor
/*   OpticalAlignInfo ( OpticalAlignInfo& rhs ); */
/*   OpticalAlignInfo ( const OpticalAlignInfo& rhs ); */

 public:
  OpticalAlignParam x_, y_, z_, angx_, angy_, angz_;
  std::vector<OpticalAlignParam> extraEntries_;
  std::string objectType_;
  std::string objectName_;
  std::string parentObjectName_;
  unsigned long objectID_;
/*   void clear() { */
/*     x_.clear(); */
/*     y_.clear(); */
/*     z_.clear(); */
/*     angx_.clear(); */
/*     angy_.clear(); */
/*     angz_.clear(); */
/*     std::cout <<"before clear extraEntries_.size() = " << extraEntries_.size() << std::endl; */
/*     extraEntries_.clear(); */
/*     std::cout <<"after clear extraEntries_.size() = " << extraEntries_.size() << std::endl; */
/*     parentObjectName_.clear(); */
/*     objectType_.clear(); */
/*     objectID_ = 0; */
/*   } */
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
