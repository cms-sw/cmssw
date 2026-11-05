/** \file
 *
 *  $Date: 2024/12/02 19:38:24 $
 *  $Revision: 1.0 $
 *  \author Pablo Martinez Ruiz del Arbol - IFCA
 */

#include <string>
#include <iostream>
#include <sstream>
#include <algorithm>

// Framework
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// Alignment

#include "Alignment/MTDAlignment/interface/MTDScenarioBuilder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Alignment/CommonAlignment/interface/Alignable.h"

//__________________________________________________________________________________________________
MTDScenarioBuilder::MTDScenarioBuilder(Alignable* alignable)
    :  // MTD alignable IDs are (currently) independent of the geometry
      MisalignmentScenarioBuilder(AlignableObjectId::Geometry::PhaseII) {
  theAlignableMTD = dynamic_cast<AlignableMTD*>(alignable);

  if (!theAlignableMTD)
    throw cms::Exception("TypeMismatch") << "Argument is not an AlignableMTD";
}

//__________________________________________________________________________________________________
void MTDScenarioBuilder::applyScenario(const edm::ParameterSet& scenario) {
  // Apply the scenario to all main components of MTD
  theScenario = scenario;
  theModifierCounter = 0;

  // Seed is set at top-level, and is mandatory
  if (this->hasParameter_("seed", theScenario))
    theModifier.setSeed(static_cast<long>(theScenario.getParameter<int>("seed")));
  else
    throw cms::Exception("BadConfig") << "No generator seed defined!";

  // BTL Barrel
  const auto& btlBarrel = theAlignableMTD->BTLBarrel();
  this->decodeMovements_(theScenario, btlBarrel, "BTL");
  // Endcap
  const auto& etlEndcaps = theAlignableMTD->ETLEndcaps();
  this->decodeMovements_(theScenario, etlEndcaps, "ETLEndcap");

  //this->moveMTD(theScenario);
  edm::LogInfo("MTDScenarioBuilder") << "Applied modifications to " << theModifierCounter << " alignables";
}

align::Scalars MTDScenarioBuilder::extractParameters(const edm::ParameterSet& pSet, const char* blockId) {
  double scale_ = 0, scaleError_ = 0, phiX_ = 0, phiY_ = 0, phiZ_ = 0;
  double dX_ = 0, dY_ = 0, dZ_ = 0;
  std::string distribution_;
  std::ostringstream error;
  edm::ParameterSet Parameters = this->getParameterSet_((std::string)blockId, pSet);
  std::vector<std::string> parameterNames = Parameters.getParameterNames();
  for (std::vector<std::string>::iterator iParam = parameterNames.begin(); iParam != parameterNames.end(); iParam++) {
    if ((*iParam) == "scale")
      scale_ = Parameters.getParameter<double>(*iParam);
    else if ((*iParam) == "distribution")
      distribution_ = Parameters.getParameter<std::string>(*iParam);
    else if ((*iParam) == "scaleError")
      scaleError_ = Parameters.getParameter<double>(*iParam);
    else if ((*iParam) == "phiX")
      phiX_ = Parameters.getParameter<double>(*iParam);
    else if ((*iParam) == "phiY")
      phiY_ = Parameters.getParameter<double>(*iParam);
    else if ((*iParam) == "phiZ")
      phiZ_ = Parameters.getParameter<double>(*iParam);
    else if ((*iParam) == "dX")
      dX_ = Parameters.getParameter<double>(*iParam);
    else if ((*iParam) == "dY")
      dY_ = Parameters.getParameter<double>(*iParam);
    else if ((*iParam) == "dZ")
      dZ_ = Parameters.getParameter<double>(*iParam);
    else if (Parameters.retrieve(*iParam).typeCode() != 'P') {  // Add unknown parameter to list
      if (error.str().empty())
        error << "Unknown parameter name(s): ";
      error << " " << *iParam;
    }
  }
  align::Scalars param;
  param.push_back(scale_);
  param.push_back(scaleError_);
  param.push_back(phiX_);
  param.push_back(phiY_);
  param.push_back(phiZ_);
  param.push_back(dX_);
  param.push_back(dY_);
  param.push_back(dZ_);
  if (distribution_ == "gaussian")
    param.push_back(0);
  else if (distribution_ == "flat")
    param.push_back(1);
  else if (distribution_ == "fix")
    param.push_back(2);

  return param;
}

void MTDScenarioBuilder::moveMTD(const edm::ParameterSet& pSet) {
  const auto& BTLbarrel = theAlignableMTD->BTLBarrel();
  const auto& ETLendcaps = theAlignableMTD->ETLEndcaps();
  //Take Parameters
  align::Scalars param = this->extractParameters(pSet, "MTD");
  double scale_ = param[0];
  double scaleError_ = param[1];
  double phiX_ = param[2];
  double phiY_ = param[3];
  double phiZ_ = param[4];
  double dX_ = param[5];
  double dY_ = param[6];
  double dZ_ = param[7];
  double dist_ = param[8];
  double dx = scale_ * dX_;
  double dy = scale_ * dY_;
  double dz = scale_ * dZ_;
  double phix = scale_ * phiX_;
  double phiy = scale_ * phiY_;
  double phiz = scale_ * phiZ_;
  double errorx = scaleError_ * dX_;
  double errory = scaleError_ * dY_;
  double errorz = scaleError_ * dZ_;
  double errorphix = scaleError_ * phiX_;
  double errorphiy = scaleError_ * phiY_;
  double errorphiz = scaleError_ * phiZ_;
  //Create an index for the chambers in the alignable vector
  align::Scalars disp;
  align::Scalars rotation;
  if (dist_ == 0) {
    const std::vector<float> disp_ = theMTDModifier.gaussianRandomVector(dx, dy, dz);
    const std::vector<float> rotation_ = theMTDModifier.gaussianRandomVector(phix, phiy, phiz);
    disp.push_back(disp_[0]);
    disp.push_back(disp_[1]);
    disp.push_back(disp_[2]);
    rotation.push_back(rotation_[0]);
    rotation.push_back(rotation_[1]);
    rotation.push_back(rotation_[2]);
  } else if (dist_ == 1) {
    const std::vector<float> disp_ = theMTDModifier.flatRandomVector(dx, dy, dz);
    const std::vector<float> rotation_ = theMTDModifier.flatRandomVector(phix, phiy, phiz);
    disp.push_back(disp_[0]);
    disp.push_back(disp_[1]);
    disp.push_back(disp_[2]);
    rotation.push_back(rotation_[0]);
    rotation.push_back(rotation_[1]);
    rotation.push_back(rotation_[2]);
  } else {
    disp.push_back(dx);
    disp.push_back(dy);
    disp.push_back(dz);
    rotation.push_back(phix);
    rotation.push_back(phiy);
    rotation.push_back(phiz);
  }
  for (const auto& iter : BTLbarrel) {
    theMTDModifier.moveAlignable(iter, false, true, disp[0], disp[1], disp[2]);
    theMTDModifier.rotateAlignable(iter, false, true, rotation[0], rotation[1], rotation[2]);
    theMTDModifier.addAlignmentPositionError(iter, errorx, errory, errorz);
    theMTDModifier.addAlignmentPositionErrorFromRotation(iter, errorphix, errorphiy, errorphiz);
  }
  for (const auto& iter : ETLendcaps) {
    theMTDModifier.moveAlignable(iter, false, true, disp[0], disp[1], disp[2]);
    theMTDModifier.rotateAlignable(iter, false, true, rotation[0], rotation[1], rotation[2]);
    theMTDModifier.addAlignmentPositionError(iter, errorx, errory, errorz);
    theMTDModifier.addAlignmentPositionErrorFromRotation(iter, errorphix, errorphiy, errorphiz);
  }
}
