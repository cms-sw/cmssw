/**
 * \file PedeReader.cc
 *
 *  \author    : Gero Flucke
 *  date       : November 2006
 *  $Revision: 1.2 $
 *  $Date: 2006/11/07 10:45:09 $
 *  (last update by $Author: flucke $)
 */

#include "PedeReader.h"
#include "PedeSteerer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/AlignableDet.h"
#include "Alignment/CommonAlignment/interface/AlignmentParameters.h"

#include "Alignment/CommonAlignmentParametrization/interface/RigidBodyAlignmentParameters.h"
#include "Alignment/CommonAlignmentParametrization/interface/CompositeRigidBodyAlignmentParameters.h"

#include "Alignment/MillePedeAlignmentAlgorithm/interface/MillePedeVariables.h"

#include <map>


const unsigned int PedeReader::myMaxNumValPerParam = 5;

//__________________________________________________________________________________________________
PedeReader::PedeReader(const char *pedeResultFile, const PedeSteerer &steerer) 
  : myPedeResult(pedeResultFile, std::ios::in), mySteerer(steerer)
{
  if (!myPedeResult.is_open()) {
    edm::LogError("Alignmnet") << "Problem opening pede output file " << pedeResultFile;
  }

}

//__________________________________________________________________________________________________
bool PedeReader::read(std::vector<Alignable*> &alignables)
{
  alignables.clear();
  myPedeResult.seekg(0, std::ios::beg); // back to start
  bool isAllOk = true;

  std::map<Alignable*,Alignable*> uniqueList;
  
  // loop on lines of text file
  unsigned int nParam = 0;
  while (myPedeResult.good() && !myPedeResult.eof()) {
    // read label
    unsigned int paramLabel = 0;
    if (!this->readIfSameLine<unsigned int>(myPedeResult, paramLabel)) continue; // empty line?

    // read up to maximal number of pede result per parameter
    float buffer[myMaxNumValPerParam] = {0.};
    unsigned int bufferPos = 0;
    for ( ; bufferPos < myMaxNumValPerParam; ++bufferPos) {
      if (!this->readIfSameLine<float>(myPedeResult, buffer[bufferPos])) break;
    }

    Alignable *alignable = this->setParameter(paramLabel, bufferPos, buffer);
    if (!alignable) {
      isAllOk = false;  // or error?
      continue;
    }
    uniqueList[alignable] = alignable;
    ++nParam;
  }

  // add Alignables to output
  for ( std::map<Alignable*,Alignable*>::const_iterator iAli = uniqueList.begin();
        iAli != uniqueList.end(); ++iAli) {
    alignables.push_back((*iAli).first);
  }

  edm::LogInfo("Alignment") << "@SUB=PedeReader::read" << nParam << " parameters for "
                            << alignables.size() << " alignables";

  return isAllOk;
}


//__________________________________________________________________________________________________
template<class T> 
bool PedeReader::readIfSameLine(std::ifstream &aStream, T &outValue) const
{

  while (true) {
    const int aChar = aStream.get();
    if (!aStream.good()) return false;

    switch(aChar) {
    case ' ':
    case '\t':
      continue; // to next character
    case '\n':
      return false; // end of line
    default:
      aStream.unget();
      aStream >> outValue;
      if (aStream.fail()) return false; // NOT if (!aStream.good())
      else                return true;
    } // switch
  } // while

  edm::LogError("Alignment") << "@SUB=PedeReader::readIfSameLine" << "Should never come here!";
  return false;
}

#include <iostream>

//__________________________________________________________________________________________________
Alignable* PedeReader::setParameter(unsigned int paramLabel,
                                    unsigned int bufLength, float *buf) const
{
  Alignable *alignable = mySteerer.alignableFromLabel(paramLabel);
  if (alignable) {
    AlignmentParameters *params = this->checkAliParams(alignable);
    MillePedeVariables *userParams = // static cast ensured by previous checkAliParams
      static_cast<MillePedeVariables*>(params->userVariables());
    
    AlgebraicVector parVec(params->parameters());
    AlgebraicSymMatrix covMat(params->covariance());
    const unsigned int paramNum = mySteerer.paramNumFromLabel(paramLabel);
    userParams->setAllDefault(paramNum);

    switch (bufLength) {
    case 5:
      userParams->globalCor()[paramNum] = buf[4];
      // no break
    case 4:
      covMat[paramNum][paramNum] = buf[3]*buf[3];
      // no break
    case 3:
      userParams->diffBefore()[paramNum] = buf[2];
      // no break
    case 2: 
      parVec[paramNum] = buf[0];
      userParams->preSigma()[paramNum] = buf[1]; // probably means fixed
      if (!userParams->isFixed(paramNum) && bufLength == 2) {
        edm::LogError("Alignment") << "@SUB=PedeReader::setParameter"
                                   << "Param " << paramLabel << " (from "
                                   << typeid(*alignable).name() << ") without result!";
        userParams->isValid()[paramNum] = false;
      }
      break;
    case 0:
    case 1:
    default:
      edm::LogError("Alignment") << "@SUB=PedeReader::setParameter"
                                 << "expect 2 to 5 values, got " << bufLength 
                                 << " for label " << paramLabel;
      break;
    }
    alignable->setAlignmentParameters(params->clone(parVec, covMat));//transferred mem. responsib.
  }
  return alignable;
}

//__________________________________________________________________________________________________
AlignmentParameters* PedeReader::checkAliParams(Alignable *alignable) const
{
  // first check that we have parameters
  AlignmentParameters *params = alignable->alignmentParameters();
  if (!params) {
    const AlgebraicVector par(RigidBodyAlignmentParameters::N_PARAM, 0);
    const AlgebraicSymMatrix cov(RigidBodyAlignmentParameters::N_PARAM, 0);
    
    bool isHigherLevel = false;
    AlignableDet *alidet = dynamic_cast<AlignableDet*>(alignable);
    if (alidet != 0) { // alignable Det
      params = new RigidBodyAlignmentParameters(alignable, par, cov);
    } else { // higher level object
      params = new CompositeRigidBodyAlignmentParameters(alignable, par, cov);
      isHigherLevel = true;
    }
    edm::LogInfo("Alignment") << "@SUB=PedeReader::checkAliParams"
                              << "build " << (isHigherLevel ? "Composite" : "" ) 
                              << "RigidBodyAlignmentParameters for alignable with label " 
                              << mySteerer.alignableLabel(alignable);
    alignable->setAlignmentParameters(params); // transferred memory responsibility
  }
  
  // now check that we have user parameters of correct type:
  if (!dynamic_cast<MillePedeVariables*>(params->userVariables())) {
    edm::LogInfo("Alignment") << "@SUB=PedeReader::checkAliParams"
                              << "add user variables for alignable with label " 
                              << mySteerer.alignableLabel(alignable);
    params->setUserVariables(new MillePedeVariables(params->size()));
  }
  
  return params;
}
