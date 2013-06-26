#include "CommonTools/Statistics/interface/ChiSquared.h"
#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"


float ChiSquared::value() const
{
  return theValue;
}


float ChiSquared::degreesOfFreedom() const
{
  return theNDF;
}


float ChiSquared::probability() const
{
  return ChiSquaredProbability(value(), degreesOfFreedom());
}


float ChiSquared::lnProbability() const
{
  return LnChiSquaredProbability(value(), degreesOfFreedom());
}
