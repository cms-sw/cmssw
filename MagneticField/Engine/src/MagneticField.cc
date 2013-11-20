/** \file
 *
 *  \author N. Amapane - CERN
 */

#include "MagneticField/Engine/interface/MagneticField.h"

MagneticField::MagneticField() { nominalValueCompiuted.store(kUnset, std::memory_order_release); }

MagneticField::MagneticField(const MagneticField& orig) : theNominalValue (orig.theNominalValue)
{ nominalValueCompiuted.store(orig.nominalValueCompiuted.load(std::memory_order_acquire), std::memory_order_release); }

MagneticField::~MagneticField(){}

int MagneticField::computeNominalValue() const {
  return int((inTesla(GlobalPoint(0.f,0.f,0.f))).z() * 10.f + 0.5f);
}

int MagneticField::nominalValue() const {
  if(kSet==nominalValueCompiuted.load(std::memory_order_acquire)) return theNominalValue;

  //need to make one
  int tmp = computeNominalValue();

  //Try to cache
  char expected = kUnset;
  if(nominalValueCompiuted.compare_exchange_strong(expected, kSetting) ) {
    //it is our job to set the value
    std::swap(theNominalValue,tmp);

    //this must be after the swap
    nominalValueCompiuted.store(kSet, std::memory_order_release);
    return theNominalValue;
  }
  //another thread beat us to trying to set theNominalValue
  // since we don't know when the other thread will finish
  // we just return tmp
  return tmp;
}
