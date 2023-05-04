#include "L1Trigger/CSCTriggerPrimitives/interface/CSCBaseboard.h"

CSCBaseboard::Parameters::Parameters(const edm::ParameterSet& conf)
    : conf_(&conf),
      commonParams_(conf.getParameter<edm::ParameterSet>("commonParam")),
      showerParams_(conf.getParameterSet("showerParam")){};

void CSCBaseboard::Parameters::chooseParams(std::string_view tmb, std::string_view alct, std::string_view clct) {
  if (tmbName_ != tmb) {
    tmbParams_ = conf_->getParameter<edm::ParameterSet>(std::string(tmb));
    tmbName_ = tmb;
  }
  if (alctName_ != alct) {
    alctParams_ = conf_->getParameter<edm::ParameterSet>(std::string(alct));
    alctName_ = alct;
  }
  if (clctName_ != clct) {
    clctParams_ = conf_->getParameter<edm::ParameterSet>(std::string(clct));
    clctName_ = clct;
  }
}

CSCBaseboard::CSCBaseboard(
    unsigned endcap, unsigned station, unsigned sector, unsigned subsector, unsigned chamber, Parameters& conf)
    : theEndcap(endcap), theStation(station), theSector(sector), theSubsector(subsector), theTrigChamber(chamber) {
  theRegion = (theEndcap == 1) ? 1 : -1;

  theRing = CSCTriggerNumbering::ringFromTriggerLabels(theStation, theTrigChamber);

  theChamber = CSCTriggerNumbering::chamberFromTriggerLabels(theSector, theSubsector, theStation, theTrigChamber);
  isME11_ = (theStation == 1 && theRing == 1);
  isME21_ = (theStation == 2 && theRing == 1);
  isME31_ = (theStation == 3 && theRing == 1);
  isME41_ = (theStation == 4 && theRing == 1);
  isME12_ = (theStation == 1 && theRing == 2);
  isME22_ = (theStation == 2 && theRing == 2);
  isME32_ = (theStation == 3 && theRing == 2);
  isME42_ = (theStation == 4 && theRing == 2);
  isME13_ = (theStation == 1 && theRing == 3);

  const bool hasTMB(isME12_ or isME22_ or isME32_ or isME42_ or isME13_);
  const bool hasOTMB(isME11_ or isME21_ or isME31_ or isME41_);
  cscId_ = CSCDetId(theEndcap, theStation, theRing, theChamber, 0);

  theCSCName_ = CSCDetId::chamberName(theEndcap, theStation, theRing, theChamber);

  runPhase2_ = conf.commonParams().getParameter<bool>("runPhase2");

  enableAlctPhase2_ = conf.commonParams().getParameter<bool>("enableAlctPhase2");

  disableME1a_ = conf.commonParams().getParameter<bool>("disableME1a");

  gangedME1a_ = conf.commonParams().getParameter<bool>("gangedME1a");

  runME11Up_ = conf.commonParams().getParameter<bool>("runME11Up");
  runME21Up_ = conf.commonParams().getParameter<bool>("runME21Up");
  runME31Up_ = conf.commonParams().getParameter<bool>("runME31Up");
  runME41Up_ = conf.commonParams().getParameter<bool>("runME41Up");

  runME11ILT_ = conf.commonParams().getParameter<bool>("runME11ILT");
  runME21ILT_ = conf.commonParams().getParameter<bool>("runME21ILT");

  run3_ = conf.commonParams().getParameter<bool>("run3");
  runCCLUT_TMB_ = conf.commonParams().getParameter<bool>("runCCLUT_TMB");
  runCCLUT_OTMB_ = conf.commonParams().getParameter<bool>("runCCLUT_OTMB");
  // check if CCLUT should be on in this chamber
  runCCLUT_ = (hasTMB and runCCLUT_TMB_) or (hasOTMB and runCCLUT_OTMB_);

  // general case
  std::string_view tmbParams = "tmbPhase1";
  std::string_view alctParams = "alctPhase1";
  std::string_view clctParams = "clctPhase1";

  const bool upgradeME11 = runPhase2_ and isME11_ and runME11Up_;
  const bool upgradeME21 = runPhase2_ and isME21_ and runME21Up_;
  const bool upgradeME31 = runPhase2_ and isME31_ and runME31Up_;
  const bool upgradeME41 = runPhase2_ and isME41_ and runME41Up_;
  const bool upgradeME = upgradeME11 or upgradeME21 or upgradeME31 or upgradeME41;

  if (upgradeME) {
    tmbParams = "tmbPhase2";
    clctParams = "clctPhase2";
    // upgrade ME1/1
    if (upgradeME11) {
      // do not run the Phase-2 ALCT for Run-3
      if (enableAlctPhase2_) {
        alctParams = "alctPhase2";
      }

      if (runME11ILT_) {
        tmbParams = "tmbPhase2GE11";
        clctParams = "clctPhase2GEM";
      }
    }
    // upgrade ME2/1
    if (upgradeME21 and runME21ILT_) {
      tmbParams = "tmbPhase2GE21";
      clctParams = "clctPhase2GEM";
      alctParams = "alctPhase2GEM";
    }
  }
  conf.chooseParams(tmbParams, alctParams, clctParams);
}

CSCBaseboard::CSCBaseboard() : theEndcap(1), theStation(1), theSector(1), theSubsector(1), theTrigChamber(1) {
  theRing = 1;
  theChamber = 1;
  runPhase2_ = false;
  disableME1a_ = false;
  gangedME1a_ = false;
}

CSCChamber const* CSCBaseboard::cscChamber(const CSCGeometry& g) const { return g.chamber(cscId_); }

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
