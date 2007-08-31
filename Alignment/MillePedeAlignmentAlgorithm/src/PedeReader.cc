/**
 * \file PedeReader.cc
 *
 *  \author    : Gero Flucke
 *  date       : November 2006
 *  $Revision: 1.6 $
 *  $Date: 2007/07/12 17:32:39 $
 *  (last update by $Author: flucke $)
 */

#include "PedeReader.h"
#include "PedeSteerer.h"
#include "PedeLabeler.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/AlignableDet.h"
#include "Alignment/CommonAlignment/interface/AlignmentParameters.h"

#include "Alignment/CommonAlignmentParametrization/interface/RigidBodyAlignmentParameters.h"

#include "Alignment/MillePedeAlignmentAlgorithm/interface/MillePedeVariables.h"

#include <map>
#include <string>


const unsigned int PedeReader::myMaxNumValPerParam = 5;

//__________________________________________________________________________________________________
PedeReader::PedeReader(const edm::ParameterSet &config, const PedeSteerer &steerer) 
  : mySteerer(steerer)
{
  std::string pedeResultFile(config.getUntrackedParameter<std::string>("fileDir"));
  if (pedeResultFile.empty()) pedeResultFile = steerer.directory(); // includes final '/'
  else if (pedeResultFile.find_last_of('/') != pedeResultFile.size() - 1) {
    pedeResultFile += '/'; // directory may need '/'
  }

  pedeResultFile += config.getParameter<std::string>("readFile");
  myPedeResult.open(pedeResultFile.c_str(), std::ios::in);
  if (!myPedeResult.is_open()) {
    edm::LogError("Alignment") << "@SUB=PedeReader"
                               << "Problem opening pede output file " << pedeResultFile;
  }
}

//__________________________________________________________________________________________________
bool PedeReader::read(std::vector<Alignable*> &alignables)
{
  alignables.clear();
  myPedeResult.seekg(0, std::ios::beg); // back to start
  bool isAllOk = true;

  std::map<Alignable*,Alignable*> uniqueList; // Probably should use a std::set here...
  
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

  return isAllOk && nParam; // nParam == 0: empty or bad file
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
      if (aStream.fail()) {// not correct type 'T' (!aStream.good() is true also in case of EOF)
        aStream.clear();
        while (aStream.good() && aStream.get() != '\n'); // forward to end of line
        return false; 
      } else {
        return true;
      }
    } // switch
  } // while

  edm::LogError("Alignment") << "@SUB=PedeReader::readIfSameLine" << "Should never come here!";
  return false;
}

//__________________________________________________________________________________________________
Alignable* PedeReader::setParameter(unsigned int paramLabel,
                                    unsigned int bufLength, float *buf) const
{
  Alignable *alignable = mySteerer.labels().alignableFromLabel(paramLabel);
  if (alignable) {
    AlignmentParameters *params = this->checkAliParams(alignable);
    MillePedeVariables *userParams = // static cast ensured by previous checkAliParams
      static_cast<MillePedeVariables*>(params->userVariables());
    // might overwrite (?):
    userParams->setLabel(mySteerer.labels().alignableLabelFromLabel(paramLabel));

    AlgebraicVector parVec(params->parameters());
    AlgebraicSymMatrix covMat(params->covariance());
    const unsigned int paramNum = mySteerer.labels().paramNumFromLabel(paramLabel);

    userParams->setAllDefault(paramNum);
    const double cmsToPede = mySteerer.cmsToPedeFactor(paramNum);

    switch (bufLength) {
    case 5: // global correlation
      userParams->globalCor()[paramNum] = buf[4]; // no break
    case 4: // uncertainty
      userParams->sigma()[paramNum] = buf[3] / cmsToPede;
      covMat[paramNum][paramNum] = userParams->sigma()[paramNum] * userParams->sigma()[paramNum];
      // no break;
    case 3: // difference to start value
      userParams->diffBefore()[paramNum] = buf[2] / cmsToPede; // no break
    case 2: 
      parVec[paramNum] = buf[0] / cmsToPede * mySteerer.parameterSign(); // parameter
      userParams->parameter()[paramNum] = parVec[paramNum]; // duplicate in millepede parameters
      userParams->preSigma()[paramNum] = buf[1];  // presigma given, probably means fixed
      if (!userParams->isFixed(paramNum)) {
        userParams->preSigma()[paramNum] /= cmsToPede;
        if (bufLength == 2) {
          edm::LogError("Alignment") << "@SUB=PedeReader::setParameter"
                                     << "Param " << paramLabel << " (from "
                                     << typeid(*alignable).name() << ") without result!";
          userParams->isValid()[paramNum] = false;
        }
      }
      break;
    case 0:
    case 1:
    default:
      edm::LogError("Alignment") << "@SUB=PedeReader::setParameter"
                                 << "Expect 2 to 5 values, got " << bufLength 
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
    // How to check in future what kind of parameters are needed?
    params = new RigidBodyAlignmentParameters(alignable, false);

    edm::LogInfo("Alignment") << "@SUB=PedeReader::checkAliParams"
                              << "Build RigidBodyAlignmentParameters for alignable with label "
                              << mySteerer.labels().alignableLabel(alignable);
    alignable->setAlignmentParameters(params); // transferred memory responsibility
  }
  
  // now check that we have user parameters of correct type:
  if (!dynamic_cast<MillePedeVariables*>(params->userVariables())) {
    edm::LogInfo("Alignment") << "@SUB=PedeReader::checkAliParams"
                              << "Add user variables for alignable with label " 
                              << mySteerer.labels().alignableLabel(alignable);
    params->setUserVariables(new MillePedeVariables(params->size()));
  }
  
  return params;
}
