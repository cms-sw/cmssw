#include "L1Trigger/CSCTriggerPrimitives/src/CSCBaseboard.h"

CSCBaseboard::CSCBaseboard(unsigned endcap, unsigned station,
                           unsigned sector, unsigned subsector,
                           unsigned chamber,
                           const edm::ParameterSet& conf) :
  theEndcap(endcap),
  theStation(station),
  theSector(sector),
  theSubsector(subsector),
  theTrigChamber(chamber)
{
  theRegion = (theEndcap == 1) ? 1: -1;

  // Ring number
  theRing = CSCTriggerNumbering::ringFromTriggerLabels(theStation, theTrigChamber);

  // actual chamber number
  theChamber = CSCTriggerNumbering::chamberFromTriggerLabels(theSector, theSubsector,
                                                             theStation, theTrigChamber);
  // is this an ME11 chamber?
  isME11_ = (theStation == 1 && theRing == 1);

  // CSCDetId for this chamber
  cscId_ = CSCDetId(theEndcap, theStation, theRing, theChamber, 0);

  // Parameters common for all boards
  commonParams_ = conf.getParameter<edm::ParameterSet>("commonParam");

  // Flag for SLHC studies
  isSLHC_       = commonParams_.getParameter<bool>("isSLHC");

  // run the upgrade for the Phase-II ME1/1 integrated local trigger
  runME11ILT_ = commonParams_.existsAs<bool>("runME11ILT") ?
    commonParams_.getParameter<bool>("runME11ILT"):false;

  // run the upgrade for the Phase-II ME2/1 integrated local trigger
  runME21ILT_ = commonParams_.existsAs<bool>("runME21ILT")?
    commonParams_.getParameter<bool>("runME21ILT"):false;

  // run the upgrade for the Phase-II ME3/1-ME4/1 local trigger
  runME3141ILT_ = commonParams_.existsAs<bool>("runME3141ILT")?
    commonParams_.getParameter<bool>("runME3141ILT"):false;

  // chamber name, e.g. ME+1/1/9
  theCSCName_ = CSCDetId::chamberName(theEndcap, theStation, theRing, theChamber);

  upgradeChambers_ = commonParams_.existsAs< std::vector<std::string> >("upgradeChambers") ?
    commonParams_.getParameter< std::vector<std::string> >("upgradeChambers"):std::vector<std::string>();

  // is this particular board (ALCT processor, TMB or CLCT processor) running the upgrade algorithm?
  runUpgradeBoard_ = false;
  if (isSLHC_ and std::find(upgradeChambers_.begin(), upgradeChambers_.end(), theCSCName_) != upgradeChambers_.end()){
    runUpgradeBoard_ = true;
  }

  // run upgrade scenarios for all MEX/1 stations
  if (isSLHC_ and theRing == 1 and runUpgradeBoard_){
    if (theStation == 1) {
      tmbParams_ = conf.getParameter<edm::ParameterSet>("tmbSLHC");
      alctParams_ = conf.getParameter<edm::ParameterSet>("alctParam07");
      clctParams_ = conf.getParameter<edm::ParameterSet>("clctSLHC");
      if (runME11ILT_) {
        tmbParams_ = conf.getParameter<edm::ParameterSet>("me11tmbSLHCGEM");
      }
    }
    else if (theStation == 2 and runME21ILT_) {
      tmbParams_ = conf.getParameter<edm::ParameterSet>("me21tmbSLHCGEM");
      alctParams_ = conf.getParameter<edm::ParameterSet>("alctSLHCME21");
      clctParams_ = conf.getParameter<edm::ParameterSet>("clctSLHCME21");
    }
    else if ((theStation == 3 or theStation == 4) and runME3141ILT_) {
      tmbParams_ = conf.getParameter<edm::ParameterSet>("me3141tmbSLHC");
      alctParams_ = conf.getParameter<edm::ParameterSet>("alctSLHCME3141");
      clctParams_ = conf.getParameter<edm::ParameterSet>("clctSLHCME3141");
    }
  } else {
    tmbParams_ = conf.getParameter<edm::ParameterSet>("tmbParam");
    alctParams_ = conf.getParameter<edm::ParameterSet>("alctParam07");
    clctParams_ = conf.getParameter<edm::ParameterSet>("clctParam07");
  }

  // special configuration parameters for ME11 treatment
  disableME1a_ = commonParams_.getParameter<bool>("disableME1a");

  // special configuration parameters for ME11 treatment
  gangedME1a_ = commonParams_.getParameter<bool>("gangedME1a");
}

CSCBaseboard::CSCBaseboard() :
                   theEndcap(1), theStation(1), theSector(1),
                   theSubsector(1), theTrigChamber(1)
{
  theRing = 1;
  theChamber = 1;
  isSLHC_ = false;
  disableME1a_ = false;
  gangedME1a_ = false;
}

void CSCBaseboard::setCSCGeometry(const CSCGeometry *g)
{
  cscGeometry_ = g;
  cscChamber_ = cscGeometry_->chamber(cscId_);
}
