#ifndef L1Trigger_CSCTriggerPrimitives_CSCBaseboard_h
#define L1Trigger_CSCTriggerPrimitives_CSCBaseboard_h

#include "DataFormats/MuonDetId/interface/CSCTriggerNumbering.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "L1Trigger/CSCCommonTrigger/interface/CSCConstants.h"
#include "L1Trigger/CSCTriggerPrimitives/src/CSCUpgradeMotherboardLUT.h"
#include "L1Trigger/CSCTriggerPrimitives/src/CSCUpgradeMotherboardLUTGenerator.h"

class CSCBaseboard
{
 public:
  /** Normal constructor. */
  CSCBaseboard(unsigned endcap, unsigned station, unsigned sector,
                 unsigned subsector, unsigned chamber,
                 const edm::ParameterSet& conf);

  /** Constructor for use during testing. */
  CSCBaseboard();

  /** Default destructor. */
  virtual ~CSCBaseboard() = default;

  void setCSCGeometry(const CSCGeometry *g);

 protected:
  // Parameters common for all boards
  edm::ParameterSet commonParams_;

  // Motherboard parameters:
  edm::ParameterSet tmbParams_;

  // Motherboard parameters:
  edm::ParameterSet alctParams_;

  // Motherboard parameters:
  edm::ParameterSet clctParams_;

  /** Chamber id (trigger-type labels). */
  const unsigned theEndcap;
  const unsigned theStation;
  const unsigned theSector;
  const unsigned theSubsector;
  const unsigned theTrigChamber;
  unsigned theRegion;
  unsigned theRing;
  unsigned theChamber;

  bool isME11_;

  CSCDetId cscId_;
  const CSCGeometry* cscGeometry_;
  const CSCChamber* cscChamber_;

  std::vector<std::string> upgradeChambers_;
  std::string theCSCName_;
  bool runUpgradeBoard_;

  /** Flag for SLHC studies. */
  bool isSLHC_;

  /** SLHC: special configuration parameters for ME1a treatment */
  bool disableME1a_, gangedME1a_;

  /** SLHC: run the upgrade for the Phase-II ME1/1 integrated local trigger */
  bool runME11ILT_;

  /** SLHC: run the upgrade for the Phase-II ME2/1 integrated local trigger */
  bool runME21ILT_;

  /** SLHC: run the upgrade for the Phase-II ME3/1(ME4/1) integrated local trigger */
  bool runME3141ILT_;
};
#endif
