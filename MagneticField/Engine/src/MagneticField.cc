/** \file
 *
 *  \author N. Amapane - CERN
 */

#include "MagneticField/Engine/interface/MagneticField.h"

MagneticField::MagneticField() : nominalValueCompiuted(kUnset), theNominalValue(0)
{
}

MagneticField::MagneticField(const MagneticField& orig) : nominalValueCompiuted(kUnset), theNominalValue(0)
{
  if(orig.nominalValueCompiuted.load() == kSet) {
    theNominalValue = orig.theNominalValue;
    nominalValueCompiuted.store(kSet);
  }
}

MagneticField::~MagneticField(){}

int MagneticField::computeNominalValue() const {
  return int((inTesla(GlobalPoint(0.f,0.f,0.f))).z() * 10.f + 0.5f);
}

int MagneticField::nominalValue() const {
  if(kSet==nominalValueCompiuted.load()) return theNominalValue;

  //need to make one
  int tmp = computeNominalValue();

  //Try to cache
  char expected = kUnset;
  if(nominalValueCompiuted.compare_exchange_strong(expected, kSetting) ) {
    //it is our job to set the value
    std::swap(theNominalValue,tmp);

    //this must be after the swap
    nominalValueCompiuted.store(kSet);
    return theNominalValue;
  }
  //another thread beat us to trying to set theNominalValue
  // since we don't know when the other thread will finish
  // we just return tmp
  return tmp;
}
