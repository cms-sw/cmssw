// File and Version Information:
//      $Id: SprData.hh,v 1.3 2007/11/12 06:19:15 narsky Exp $
//
// Description:
//      Class SprData :
//          Collection of SprPoints owned by SprData.
//
// Environment:
//      Software developed for the BaBar Detector at the SLAC B-Factory.
//
// Author List:
//      Ilya Narsky                     Original author
//
// Copyright Information:
//      Copyright (C) 2005              California Institute of Technology
//
//------------------------------------------------------------------------
 
#ifndef _SprData_HH
#define _SprData_HH

#include <vector>
#include <string>
#include <map>
#include <iostream>

class SprPoint;
class SprClass;


class SprData
{
public:
  ~SprData();

  SprData(bool ownPoints=true) 
    : ownPoints_(ownPoints), 
      label_(), 
      vars_(), 
      dim_(0), 
      data_()
  {}

  SprData(const char* label, 
	  const std::vector<std::string>& vars,
	  bool ownPoints=true)
    : ownPoints_(ownPoints), 
      label_(), 
      vars_(), 
      dim_(0), 
      data_()
  { this->setVars(vars); }

  // makes an empty copy of data
  SprData* emptyCopy() const;
  SprData* copy() const;

  // accessors
  SprPoint* operator[](int i) const { return data_[i]; }
  inline SprPoint* at(int i) const;
  std::string label() const { return label_; }
  unsigned dim() const { return dim_; }
  void vars(std::vector<std::string>& vars) const { vars = vars_; }
  unsigned size() const { return data_.size(); }
  bool empty() const { return data_.empty(); }
  unsigned ptsInClass(const SprClass& cls) const;
  SprPoint* find(unsigned index) const;
  int dimIndex(const char* var) const;
  bool ownPoints() const { return ownPoints_; }

  // modifiers
  SprPoint* insert(SprPoint* p);
  SprPoint* insert(int cls, const std::vector<double>& v);
  SprPoint* insert(unsigned index, int cls, const std::vector<double>& v);
  SprPoint* uncheckedInsert(SprPoint* p) {
    data_.push_back(p);
    return p;
  }
  void clear();
  void setLabel(const char* label) { label_ = label; }
  bool setVars(const std::vector<std::string>& vars);
  void setDim(unsigned dim) { dim_ = dim; }
  void setOwnPoints(bool own) { ownPoints_ = own; }

private:
  bool ownPoints_;
  std::string label_;// label
  std::vector<std::string> vars_;// variable names
  unsigned dim_;// dimensionality of space
  std::vector<SprPoint*> data_;// collection of data points
};

inline SprPoint* SprData::at(int i) const {
  if( i>=0 && i<(int)data_.size() ) return data_[i];
  std::cerr << "Index out of range for data " << i << " " 
	    << data_.size() << std::endl;
  return 0;
}

#endif




