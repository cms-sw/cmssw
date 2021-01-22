#ifndef L1Trigger_CSCTriggerPrimitives_CSCBaseboard_h
#define L1Trigger_CSCTriggerPrimitives_CSCBaseboard_h

#include "DataFormats/MuonDetId/interface/CSCTriggerNumbering.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "DataFormats/L1TMuon/interface/CSCConstants.h"
#include "L1Trigger/CSCTriggerPrimitives/interface/CSCPatternBank.h"
#include "L1Trigger/CSCTriggerPrimitives/interface/CSCUpgradeMotherboardLUT.h"
#include "CondFormats/CSCObjects/interface/CSCDBL1TPParameters.h"

class CSCBaseboard {
public:
  /** Normal constructor. */
  CSCBaseboard(unsigned endcap,
               unsigned station,
               unsigned sector,
               unsigned subsector,
               unsigned chamber,
               const edm::ParameterSet& conf);

  /** Constructor for use during testing. */
  CSCBaseboard();

  /** Default destructor. */
  virtual ~CSCBaseboard() = default;

  void setCSCGeometry(const CSCGeometry* g);

  std::string getCSCName() const { return theCSCName_; }

protected:
  void checkConfigParameters(unsigned int& var,
                             const unsigned int var_max,
                             const unsigned int var_def,
                             const std::string& var_str);

  /** Chamber id (trigger-type labels). */
  const unsigned theEndcap;
  const unsigned theStation;
  const unsigned theSector;
  const unsigned theSubsector;
  const unsigned theTrigChamber;
  unsigned theRegion;
  unsigned theRing;
  unsigned theChamber;

  // is this an ME11 chamber?
  bool isME11_;
  bool isME21_;
  bool isME31_;
  bool isME41_;

  // CSCDetId for this chamber
  CSCDetId cscId_;

  /** Verbosity level: 0: no print (default).
   *                   1: print only ALCTs found.
   *                   2: info at every step of the algorithm.
   *                   3: add special-purpose prints. */
  int infoV;

  const CSCGeometry* cscGeometry_;
  const CSCChamber* cscChamber_;

  // Parameters common for all boards
  edm::ParameterSet commonParams_;

  // Motherboard parameters:
  edm::ParameterSet tmbParams_;

  // ALCT Processor parameters:
  edm::ParameterSet alctParams_;

  // CLCT Processor parameters:
  edm::ParameterSet clctParams_;

  // chamber name, e.g. ME+1/1/9
  std::string theCSCName_;

  /** Flag for Phase2 studies. */
  bool runPhase2_;
  bool enableAlctPhase2_;

  /** Phase2: special configuration parameters for ME1a treatment */
  bool disableME1a_, gangedME1a_;

  /** Phase2: run the upgrade for the Phase-II ME1/1 integrated local trigger */
  bool runME11ILT_;

  /** Phase2: run the upgrade for the Phase-II ME2/1 integrated local trigger */
  bool runME21ILT_;

  /** Phase2: run the upgrade local trigger (without GEMs) */
  bool runME11Up_;
  bool runME21Up_;
  bool runME31Up_;
  bool runME41Up_;

  bool runCCLUT_;
};
#endif
