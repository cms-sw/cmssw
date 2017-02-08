/** \file PedeLabelerBase.cc
 *
 * Baseclass for pede labelers
 *
 *  Original author: Andreas Mussgiller, January 2011
 *
 *  $Date: 2011/02/23 16:58:34 $
 *  $Revision: 1.3 $
 *  (last update by $Author: mussgill $)
 */

#include "Alignment/MillePedeAlignmentAlgorithm/interface/PedeLabelerBase.h"

#include "Alignment/CommonAlignmentParametrization/interface/RigidBodyAlignmentParameters.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/IntegratedCalibrationBase.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

// NOTE: Changing '+14' makes older binary files unreadable...
const unsigned int PedeLabelerBase::theMaxNumParam = RigidBodyAlignmentParameters::N_PARAM + 14;
// NOTE: Changing the offset of '700000' makes older binary files unreadable...
const unsigned int PedeLabelerBase::theParamInstanceOffset = 700000;
const unsigned int PedeLabelerBase::theMinLabel = 1; // must be > 0

PedeLabelerBase::PedeLabelerBase(const TopLevelAlignables &alignables,
				 const edm::ParameterSet &config)
  :theOpenRunRange(std::make_pair<RunNumber,RunNumber>( RunNumber( cond::timeTypeSpecs[cond::runnumber].beginValue) ,  // since we know we have a runnumber here, we can
						        RunNumber( cond::timeTypeSpecs[cond::runnumber].endValue  ) )) // simply convert the Time_t to make the compiler happy
{
  
}

//___________________________________________________________________________
std::pair<IntegratedCalibrationBase*, unsigned int>
PedeLabelerBase::calibrationParamFromLabel(unsigned int label) const
{
  // Quick check whether label is in range of calibration labels:
  if (!theCalibrationLabels.empty() && label >= theCalibrationLabels.front().second) {
    // Loop on all known IntegratedCalibration's:
    for (auto iCal = theCalibrationLabels.begin(); iCal != theCalibrationLabels.end(); ++iCal) {
      if (label >= iCal->second && label < iCal->second + iCal->first->numParameters()) {
        // Label fits in range for this calibration, so return calibration
        // and subtract first label of this calibration from label.
        return std::make_pair(iCal->first, label - iCal->second);
      }
    }
    edm::LogError("LogicError") << "@SUB=PedeLabelerBase::calibrationParamFromLabel"
                                << "Label " << label << "larger than first calibration's label, "
                                << "but no calibration fits!";
  }

  // Return that nothing fits:
  return std::pair<IntegratedCalibrationBase*, unsigned int>(0,0);
}

//___________________________________________________________________________
unsigned int PedeLabelerBase::firstFreeLabel() const
{
  unsigned int nextId = this->firstNonAlignableLabel();

  for (auto iCal = theCalibrationLabels.begin(); iCal != theCalibrationLabels.end(); ++iCal) {
    nextId += iCal->first->numParameters();
  }

  return nextId;
}

//___________________________________________________________________________
unsigned int PedeLabelerBase::firstNonAlignableLabel() const
{
  
  return this->parameterInstanceOffset() * this->maxNumberOfParameterInstances() + 1;
}

//___________________________________________________________________________
unsigned int PedeLabelerBase::calibrationLabel(const IntegratedCalibrationBase* calib,
                                               unsigned int paramNum) const
{
  if (!calib) {
    throw cms::Exception("LogicError") << "PedeLabelerBase::calibrationLabel: "
                                       << "nullPtr passed!\n";
  }

  // loop on all known IntegratedCalibration's
  for (auto iCal = theCalibrationLabels.begin(); iCal != theCalibrationLabels.end(); ++iCal) {
    if (iCal->first == calib) { // found IntegratedCalibrationBase
      if (paramNum < iCal->first->numParameters()) {
        return iCal->second + paramNum;
      } else { // paramNum out of range!
        edm::LogError("LogicError")
          << "@SUB=PedeLabelerBase::calibrationLabel" << "IntegratedCalibration "
          << calib->name() << " has only " << iCal->first->numParameters()
          << " parameters, but " << paramNum << "requested!";
      }
    }
  }
  
  edm::LogError("LogicError")
    << "@SUB=PedeLabelerBase::calibrationLabel" << "IntegratedCalibration "
    << calib->name() << " not known or too few parameters.";
  
  return 0;
}

//___________________________________________________________________________
void PedeLabelerBase::addCalibrations(const std::vector<IntegratedCalibrationBase*> &iCals)
{
  unsigned int nextId = this->firstFreeLabel(); // so far next free label

  // Now foresee labels for new calibrations:
  for (auto iCal = iCals.begin(); iCal != iCals.end(); ++iCal) {
    if (*iCal) {
      theCalibrationLabels.push_back(std::make_pair(*iCal, nextId));
      nextId += (*iCal)->numParameters();
    } else {
      edm::LogError("LogicError")
        << "@SUB=PedeLabelerBase::addCalibrations" << "Ignoring nullPtr.";
    }
  }
}
