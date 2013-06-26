/**
 * \file PedeReader.cc
 *
 *  \author    : Gero Flucke
 *  date       : November 2006
 *  $Revision: 1.15 $
 *  $Date: 2013/05/29 08:52:48 $
 *  (last update by $Author: flucke $)
 */

#include "PedeReader.h"
#include "PedeSteerer.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/AlignmentParameters.h"

#include "Alignment/CommonAlignmentAlgorithm/interface/IntegratedCalibrationBase.h"

#include "Alignment/CommonAlignmentParametrization/interface/RigidBodyAlignmentParameters.h"

#include "Alignment/MillePedeAlignmentAlgorithm/interface/MillePedeVariables.h"

#include <set>
#include <string>
#include <sstream>

const unsigned int PedeReader::myMaxNumValPerParam = 5;

//__________________________________________________________________________________________________
PedeReader::PedeReader(const edm::ParameterSet &config, const PedeSteerer &steerer,
		       const PedeLabelerBase &labels, const RunRange &runrange)
  : mySteerer(steerer), myLabels(labels), myRunRange(runrange)
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
bool PedeReader::read(std::vector<Alignable*> &alignables, bool setUserVars)
{
  alignables.clear();
  myPedeResult.seekg(0, std::ios::beg); // back to start
  bool isAllOk = true;
  
  std::set<Alignable*> uniqueList;
  
  edm::LogInfo("Alignment") << "@SUB=PedeReader::read"
			    << "will read parameters for run range "
			    << myRunRange.first << " - " << myRunRange.second;
  
  // loop on lines of text file
  unsigned int nParam = 0, nParamCalib = 0, nParamUnknown = 0;
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
    
    // First check whether parameter is from any calibration (to be done before RunRange check:
    // run dependence for them is not handled here, but probably inside the calibration).
    // Double setting by calling read(..) twice for different RunRanges shouldn't harm.
    std::pair<IntegratedCalibrationBase*, unsigned int> calibParam
      = myLabels.calibrationParamFromLabel(paramLabel);
    if (calibParam.first) { // label belongs to a calibration
      if (this->setCalibrationParameter(calibParam.first, calibParam.second, bufferPos, buffer)) {
        ++nParamCalib;
      } else {
        edm::LogError("Alignment") << "@SUB=PedeReader::read" << "Problems setting results of "
                                   << "parameter " << calibParam.second << " to calibration '"
                                   << calibParam.first->name() << "' ("<< calibParam.first << ").";
        isAllOk = false;
      }
      continue; // does not belong to any Alignable, so go to next line of file 
    }
    // Now treat Alignables if paramLabel fits to run range, otherwise skip line:
    const RunRange & runRange = myLabels.runRangeFromLabel(paramLabel);
    if (!(runRange.first<=myRunRange.first && myRunRange.second<=runRange.second)) continue;
    
    Alignable *alignable = this->setParameter(paramLabel, bufferPos, buffer, setUserVars);
    if (!alignable) {
      // OK, e.g. for PedeReaderInputs: calibration parameters, but not yet known to labeler
      ++nParamUnknown;
      continue;
    }
    uniqueList.insert(alignable);
    ++nParam;
  }

  // add Alignables to output
  alignables.insert(alignables.end(), uniqueList.begin(), uniqueList.end());

  std::stringstream out; // "@SUB=PedeReader::read" cannot yet go to 'out' for proper format
  out << nParam << " parameters for " << alignables.size() << " alignables and " << nParamCalib
      << " for calibrations, " << nParamUnknown << " parameters are unknown.\n";
  if (nParamUnknown) {
    edm::LogWarning("Alignment") << "@SUB=PedeReader::read" << out.str();
  } else {
    edm::LogInfo("Alignment") << "@SUB=PedeReader::read" << out.str();
  }

  return isAllOk && (nParam + nParamCalib); // nParam+nParamCalib == 0: empty or bad file
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
                                    unsigned int bufLength, const float *buf,
                                    bool setUserVars) const
{
  Alignable *alignable = myLabels.alignableFromLabel(paramLabel);
  const unsigned int paramNum = myLabels.paramNumFromLabel(paramLabel);
  const double cmsToPede = mySteerer.cmsToPedeFactor(paramNum);
  if (alignable) {
    AlignmentParameters *params = this->checkAliParams(alignable, setUserVars);
    MillePedeVariables *userParams = // static cast ensured by previous checkAliParams
      (setUserVars ? static_cast<MillePedeVariables*>(params->userVariables()) : 0);
    // if (userParams && userParams->label() != myLabels.alignableLabelFromLabel(paramLabel)) {
    if (userParams && userParams->label() != myLabels.alignableLabel(alignable)) {
      edm::LogError("Alignment") << "@SUB=PedeReader::setParameter" 
				 << "Label mismatch: paramLabel " << paramLabel 
				 << " for alignableLabel " << userParams->label();
    }

    AlgebraicVector parVec(params->parameters());
    AlgebraicSymMatrix covMat(params->covariance());

    if (userParams) userParams->setAllDefault(paramNum);

    switch (bufLength) {
    case 5: // global correlation
      if (userParams) userParams->globalCor()[paramNum] = buf[4]; // no break
    case 4: // uncertainty
      if (userParams) userParams->sigma()[paramNum] = buf[3] / cmsToPede;
      covMat[paramNum][paramNum] = buf[3]*buf[3] / (cmsToPede*cmsToPede);
      // no break;
    case 3: // difference to start value
      if (userParams) userParams->diffBefore()[paramNum] = buf[2] / cmsToPede;
      // no break
    case 2: 
      params->setValid(true);
      parVec[paramNum] = buf[0] / cmsToPede * mySteerer.parameterSign(); // parameter
      if (userParams) {
	userParams->parameter()[paramNum] = parVec[paramNum]; // duplicate in millepede parameters
	userParams->preSigma()[paramNum] = buf[1];  // presigma given, probably means fixed
	if (!userParams->isFixed(paramNum)) {
	  userParams->preSigma()[paramNum] /= cmsToPede;
	  if (bufLength == 2) {
	    edm::LogWarning("Alignment") << "@SUB=PedeReader::setParameter"
					 << "Param " << paramLabel << " (from "
					 << typeid(*alignable).name() << ") without result!";
	    userParams->isValid()[paramNum] = false;
	    params->setValid(false);
	  }
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
  } else {
    unsigned int lasBeamId = myLabels.lasBeamIdFromLabel(paramLabel);
    edm::LogError("Alignment") << "@SUB=PedeReader::setParameter"
			       << "No alignable for paramLabel " << paramLabel
			       << ", probably LasBeam with Id " << lasBeamId
			       << ",\nparam " << paramNum << ": " 
			       << buf[0] / cmsToPede * mySteerer.parameterSign()
			       << " += " << (bufLength >= 4 ? buf[3] / cmsToPede : -99.);
  }

  return alignable;
}

//__________________________________________________________________________________________________
bool PedeReader::setCalibrationParameter(IntegratedCalibrationBase* calib, unsigned int paramNum,
                                         unsigned int bufLength, const float *buf) const
{
  if (!calib || !buf) return false;

  // FIXME: Should we attach MillePedeVariables to IntegratedCalibrationBase to store
  //        'other' results beyond value and error?
  switch (bufLength) {
  case 5: // buf[4]: global correlation - not treated yet FIXME // no break;
  case 4: // uncertainty
    calib->setParameterError(paramNum, buf[3]); // no break;
  case 3: // buf[2]: difference to start value - not treated yet // no break;
  case 2:
    if (bufLength == 2 && buf[1] >= 0.) { // buf[1]: pre-sigma, < 0 means fixed
      edm::LogWarning("Alignment") << "@SUB=PedeReader::setCalibrationParameter"
                                   << "Param " << paramNum << " of calibration '"
                                   << calib->name() << "' without result!";
    }
    return calib->setParameter(paramNum, buf[0] * mySteerer.parameterSign());
  case 0:
  case 1:
  default:
    edm::LogError("Alignment") << "@SUB=PedeReader::setCalibrationParameter"
                               << "Expect 2 to 5 values, got " << bufLength << "."; 
    return false;
  }

}

//__________________________________________________________________________________________________
AlignmentParameters* PedeReader::checkAliParams(Alignable *alignable, bool createUserVars) const
{
  // first check that we have parameters
  AlignmentParameters *params = alignable->alignmentParameters();
  if (!params) {
    throw cms::Exception("BadConfig") << "PedeReader::checkAliParams: "
				      << "Alignable without parameters.";

  }
  
  // now check that we have user parameters of correct type if requested:
  if (createUserVars && !dynamic_cast<MillePedeVariables*>(params->userVariables())) {
    edm::LogInfo("Alignment") << "@SUB=PedeReader::checkAliParams"
                              << "Add user variables for alignable with label " 
                              << myLabels.alignableLabel(alignable);
    params->setUserVariables(new MillePedeVariables(params->size(), myLabels.alignableLabel(alignable)));
  }
  
  return params;
}
