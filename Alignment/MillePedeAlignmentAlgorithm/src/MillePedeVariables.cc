/**
 * \file MillePedeVariables.cc
 *
 *
 *  \author    : Gero Flucke
 *  date       : November 2006
 *  $Revision: 1.2 $
 *  $Date: 2006/11/30 10:34:05 $
 *  (last update by $Author: flucke $)
 */

#include "Alignment/MillePedeAlignmentAlgorithm/interface/MillePedeVariables.h"

//__________________________________________________________________________________________________
MillePedeVariables::MillePedeVariables(unsigned int nParams)
  :  myIsValid(nParams), myDiffBefore(nParams), myGlobalCor(nParams), myPreSigma(nParams),
     myParameter(nParams), mySigma(nParams), myHitsX(0), myHitsY(0), myLabel(0)
{
  for (unsigned int i = 0; i < nParams; ++i) {
    this->setAllDefault(i);
  }
}

//__________________________________________________________________________________________________
bool MillePedeVariables::setAllDefault(unsigned int nParam)
{
  if (nParam >= this->size()) return false;

  myIsValid[nParam] = true;
  myDiffBefore[nParam] = -999999.;
  myGlobalCor[nParam] = -1.;
  myPreSigma[nParam] = -11.; // -1 means fixed in pede
  myParameter[nParam] = -999999.;
  mySigma[nParam] = -1.;

  return true;
}

//__________________________________________________________________________________________________
bool MillePedeVariables::isFixed(unsigned int nParam) const
{
  if (nParam >= this->size()) return false;
  
  return (this->preSigma()[nParam] < 0.);
}
