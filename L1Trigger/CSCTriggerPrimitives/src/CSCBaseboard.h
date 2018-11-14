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

  std::string getCSCName() const {return theCSCName_;}

 protected:
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

  /** Flag for SLHC studies. */
  bool isSLHC_;
  bool enableAlctSLHC_;

  /** SLHC: special configuration parameters for ME1a treatment */
  bool disableME1a_, gangedME1a_;

  // shift the BX from 7 to 8
  // the unpacked real data CLCTs have central BX at bin 7
  // however in simulation the central BX  is bin 8
  // to make a proper comparison with ALCTs we need
  // CLCT and ALCT to have the central BX in the same bin
  // this shift does not affect the readout of the CLCTs
  // emulated CLCTs put in the event should be centered at bin 7 (as in data)
  unsigned int alctClctOffset_;

  /** SLHC: run the upgrade for the Phase-II ME1/1 integrated local trigger */
  bool runME11ILT_;

  /** SLHC: run the upgrade for the Phase-II ME2/1 integrated local trigger */
  bool runME21ILT_;

  /** SLHC: run the upgrade local trigger (without GEMs) */
  bool runME11Up_;
  bool runME21Up_;
  bool runME31Up_;
  bool runME41Up_;
};
#endif
