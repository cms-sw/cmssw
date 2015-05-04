// -*- C++ -*-
#ifndef METCorrectorParameters_h
#define METCorrectorParameters_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include <string>
#include <vector>
#include <iostream>

//____________________________________________________________________________||
class METCorrectorParameters
{
public:

  METCorrectorParameters() { valid_ = false; }
  METCorrectorParameters(const std::string& fFile, const std::string& fSection = "");

  void printScreen() const;
  bool isValid() const { return valid_; }
  std::vector<double> getParVec() const {return mRecord;}

private:
  bool valid_;
  std::vector<double> mRecord;

  COND_SERIALIZABLE;
};
//____________________________________________________________________________||

#endif // METCorrectorParameters_h
