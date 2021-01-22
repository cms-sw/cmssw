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
  isME31_ = (theStation == 3 && theRing == 1);
  isME41_ = (theStation == 4 && theRing == 1);

  cscId_ = CSCDetId(theEndcap, theStation, theRing, theChamber, 0);

  commonParams_ = conf.getParameter<edm::ParameterSet>("commonParam");

  theCSCName_ = CSCDetId::chamberName(theEndcap, theStation, theRing, theChamber);

  runPhase2_ = commonParams_.getParameter<bool>("runPhase2");

  enableAlctPhase2_ = commonParams_.getParameter<bool>("enableAlctPhase2");

  disableME1a_ = commonParams_.getParameter<bool>("disableME1a");

  gangedME1a_ = commonParams_.getParameter<bool>("gangedME1a");

  runME11Up_ = commonParams_.getParameter<bool>("runME11Up");
  runME21Up_ = commonParams_.getParameter<bool>("runME21Up");
  runME31Up_ = commonParams_.getParameter<bool>("runME31Up");
  runME41Up_ = commonParams_.getParameter<bool>("runME41Up");

  runME11ILT_ = commonParams_.getParameter<bool>("runME11ILT");
  runME21ILT_ = commonParams_.getParameter<bool>("runME21ILT");

  runCCLUT_ = commonParams_.getParameter<bool>("runCCLUT");

  // general case
  tmbParams_ = conf.getParameter<edm::ParameterSet>("tmbPhase1");
  alctParams_ = conf.getParameter<edm::ParameterSet>("alctPhase1");
  clctParams_ = conf.getParameter<edm::ParameterSet>("clctPhase1");

  const bool upgradeME11 = runPhase2_ and isME11_ and runME11Up_;
  const bool upgradeME21 = runPhase2_ and isME21_ and runME21Up_;
  const bool upgradeME31 = runPhase2_ and isME31_ and runME31Up_;
  const bool upgradeME41 = runPhase2_ and isME41_ and runME41Up_;
  const bool upgradeME = upgradeME11 or upgradeME21 or upgradeME31 or upgradeME41;

  if (upgradeME) {
    tmbParams_ = conf.getParameter<edm::ParameterSet>("tmbPhase2");
    clctParams_ = conf.getParameter<edm::ParameterSet>("clctPhase2");
    // upgrade ME1/1
    if (upgradeME11) {
      // do not run the Phase-2 ALCT for Run-3
      if (enableAlctPhase2_) {
        alctParams_ = conf.getParameter<edm::ParameterSet>("alctPhase2");
      }

      if (runME11ILT_) {
        tmbParams_ = conf.getParameter<edm::ParameterSet>("tmbPhase2GE11");
        clctParams_ = conf.getParameter<edm::ParameterSet>("clctPhase2GEM");
      }
    }
    // upgrade ME2/1
    if (upgradeME21 and runME21ILT_) {
      tmbParams_ = conf.getParameter<edm::ParameterSet>("tmbPhase2GE21");
      clctParams_ = conf.getParameter<edm::ParameterSet>("clctPhase2GEM");
      alctParams_ = conf.getParameter<edm::ParameterSet>("alctPhase2GEM");
    }
  }
}

CSCBaseboard::CSCBaseboard() : theEndcap(1), theStation(1), theSector(1), theSubsector(1), theTrigChamber(1) {
  theRing = 1;
  theChamber = 1;
  runPhase2_ = false;
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
