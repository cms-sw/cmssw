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
  isME21_ = (theStation == 2 && theRing == 1);

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
      tmbParams_ = conf.getParameter<edm::ParameterSet>("tmbPhase2ME11");
      clctParams_ = conf.getParameter<edm::ParameterSet>("clctPhase2ME11");
      alctParams_ = conf.getParameter<edm::ParameterSet>("alctPhase2ME11");
      if (not enableAlctSLHC_) {
        alctParams_ = conf.getParameter<edm::ParameterSet>("alctPhase1");
      }
      if (runME11ILT_) {
        tmbParams_ = conf.getParameter<edm::ParameterSet>("tmbPhase2GE11");
        clctParams_ = conf.getParameter<edm::ParameterSet>("clctPhase2GE11");
      }
    } else if (theStation == 2 and runME21Up_) {
      tmbParams_ = conf.getParameter<edm::ParameterSet>("tmbPhase2MEX1");
      alctParams_ = conf.getParameter<edm::ParameterSet>("alctPhase2MEX1");
      clctParams_ = conf.getParameter<edm::ParameterSet>("clctPhase2MEX1");
      if (runME21ILT_) {
        tmbParams_ = conf.getParameter<edm::ParameterSet>("tmbPhase2GE21");
        clctParams_ = conf.getParameter<edm::ParameterSet>("clctPhase2GE21");
      }
    } else if ((theStation == 3 and runME31Up_) or (theStation == 4 and runME41Up_)) {
      tmbParams_ = conf.getParameter<edm::ParameterSet>("tmbPhase2MEX1");
      alctParams_ = conf.getParameter<edm::ParameterSet>("alctPhase2MEX1");
      clctParams_ = conf.getParameter<edm::ParameterSet>("clctPhase2MEX1");
    }
    //Phase2 is on but ME21, ME31, ME41 is not upgraded
    else {
      tmbParams_ = conf.getParameter<edm::ParameterSet>("tmbPhase1");
      alctParams_ = conf.getParameter<edm::ParameterSet>("alctPhase1");
      clctParams_ = conf.getParameter<edm::ParameterSet>("clctPhase1");
    }
  }
  //others
  else {
    tmbParams_ = conf.getParameter<edm::ParameterSet>("tmbPhase1");
    alctParams_ = conf.getParameter<edm::ParameterSet>("alctPhase1");
    clctParams_ = conf.getParameter<edm::ParameterSet>("clctPhase1");
  }

  use_run3_patterns_ = clctParams_.getParameter<bool>("useRun3Patterns");
  use_comparator_codes_ = clctParams_.getParameter<bool>("useComparatorCodes");
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
    edm::LogError("CSCConfigError") << "+++ Value of " + var_str + ", " << var << ", exceeds max allowed, " << var - 1
                                    << " +++\n"
                                    << "+++ Try to proceed with the default value, " + var_str + "=" << var_def
                                    << " +++\n";
    var = var_def;
  }
}
