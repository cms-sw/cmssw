/**
 * \file MillePedeVariables.cc
 *
 *
 *  \author    : Gero Flucke
 *  date       : November 2006
 *  $Revision: 1.4 $
 *  $Date: 2007/08/17 17:20:05 $
 *  (last update by $Author: flucke $)
 */

#include "Alignment/MillePedeAlignmentAlgorithm/interface/MillePedeVariables.h"

//__________________________________________________________________________________________________
MillePedeVariables::MillePedeVariables(unsigned int nParams, unsigned int label)
  :  myIsValid(nParams), myDiffBefore(nParams), myGlobalCor(nParams), myPreSigma(nParams),
     myParameter(nParams), mySigma(nParams), myHitsX(0), myHitsY(0), myLabel(label)
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
  myGlobalCor[nParam] = -.2; // -1. seems to occur also in pede output
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
