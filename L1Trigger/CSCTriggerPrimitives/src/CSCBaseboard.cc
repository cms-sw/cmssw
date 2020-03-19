#include "L1Trigger/CSCTriggerPrimitives/interface/CSCBaseboard.h"

CSCBaseboard::CSCBaseboard(unsigned endcap,
                           unsigned station,
                           unsigned sector,
                           unsigned subsector,
                           unsigned chamber,
                           const edm::ParameterSet& conf)
    : theEndcap(endcap), theStation(station), theSector(sector), theSubsector(subsector), theTrigChamber(chamber) {
  theRegion = (theEndcap == 1) ? 1 : -1;

  theRing = CSCTriggerNumbering::ringFromTriggerLabels(theStation, theTrigChamber);

  theChamber = CSCTriggerNumbering::chamberFromTriggerLabels(theSector, theSubsector, theStation, theTrigChamber);
  isME11_ = (theStation == 1 && theRing == 1);

  cscId_ = CSCDetId(theEndcap, theStation, theRing, theChamber, 0);

  commonParams_ = conf.getParameter<edm::ParameterSet>("commonParam");

  theCSCName_ = CSCDetId::chamberName(theEndcap, theStation, theRing, theChamber);

  isSLHC_ = commonParams_.getParameter<bool>("isSLHC");

  enableAlctSLHC_ = commonParams_.getParameter<bool>("enableAlctSLHC");

  disableME1a_ = commonParams_.getParameter<bool>("disableME1a");

  gangedME1a_ = commonParams_.getParameter<bool>("gangedME1a");

  alctClctOffset_ = commonParams_.getParameter<unsigned int>("alctClctOffset");

  runME11Up_ = commonParams_.getParameter<bool>("runME11Up");
  runME21Up_ = commonParams_.getParameter<bool>("runME21Up");
  runME31Up_ = commonParams_.getParameter<bool>("runME31Up");
  runME41Up_ = commonParams_.getParameter<bool>("runME41Up");

  runME11ILT_ = commonParams_.getParameter<bool>("runME11ILT");
  runME21ILT_ = commonParams_.getParameter<bool>("runME21ILT");

  if (isSLHC_ and theRing == 1) {
    if (theStation == 1 and runME11Up_) {
      tmbParams_ = conf.getParameter<edm::ParameterSet>("tmbSLHC");
      clctParams_ = conf.getParameter<edm::ParameterSet>("clctSLHC");
      alctParams_ = conf.getParameter<edm::ParameterSet>("alctSLHC");
      if (not enableAlctSLHC_) {
        alctParams_ = conf.getParameter<edm::ParameterSet>("alctParam07");
      }
      if (runME11ILT_) {
        tmbParams_ = conf.getParameter<edm::ParameterSet>("me11tmbSLHCGEM");
      }
    } else if (theStation == 2 and runME21Up_) {
      tmbParams_ = conf.getParameter<edm::ParameterSet>("meX1tmbSLHC");
      alctParams_ = conf.getParameter<edm::ParameterSet>("alctSLHCME21");
      clctParams_ = conf.getParameter<edm::ParameterSet>("clctSLHCME21");
      if (runME21ILT_) {
        tmbParams_ = conf.getParameter<edm::ParameterSet>("me21tmbSLHCGEM");
      }
    } else if ((theStation == 3 and runME31Up_) or (theStation == 4 and runME41Up_)) {
      tmbParams_ = conf.getParameter<edm::ParameterSet>("meX1tmbSLHC");
      alctParams_ = conf.getParameter<edm::ParameterSet>("alctSLHCME3141");
      clctParams_ = conf.getParameter<edm::ParameterSet>("clctSLHCME3141");
    } else {  //SLHC is on but ME21ME31ME41 is not upgraded
      tmbParams_ = conf.getParameter<edm::ParameterSet>("tmbParam");
      alctParams_ = conf.getParameter<edm::ParameterSet>("alctParam07");
      clctParams_ = conf.getParameter<edm::ParameterSet>("clctParam07");
    }
  } else {  //others
    tmbParams_ = conf.getParameter<edm::ParameterSet>("tmbParam");
    alctParams_ = conf.getParameter<edm::ParameterSet>("alctParam07");
    clctParams_ = conf.getParameter<edm::ParameterSet>("clctParam07");
  }
}

CSCBaseboard::CSCBaseboard() : theEndcap(1), theStation(1), theSector(1), theSubsector(1), theTrigChamber(1) {
  theRing = 1;
  theChamber = 1;
  isSLHC_ = false;
  disableME1a_ = false;
  gangedME1a_ = false;
}

void CSCBaseboard::setCSCGeometry(const CSCGeometry* g) {
  cscGeometry_ = g;
  cscChamber_ = cscGeometry_->chamber(cscId_);
}

void CSCBaseboard::checkConfigParameters(unsigned int& var,
                                         const unsigned int var_max,
                                         const unsigned int var_def,
                                         const std::string& var_str) {
  // Make sure that the parameter values are within the allowed range.
  if (var >= var_max) {
    if (infoV >= 0)
      edm::LogError("CSCConfigError") << "+++ Value of " + var_str + ", " << var << ", exceeds max allowed, " << var - 1
                                      << " +++\n"
                                      << "+++ Try to proceed with the default value, " + var_str + "=" << var_def
                                      << " +++\n";
    var = var_def;
  }
}
