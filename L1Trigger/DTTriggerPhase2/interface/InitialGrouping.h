#ifndef Phase2L1Trigger_DTTrigger_InitialGrouping_h
#define Phase2L1Trigger_DTTrigger_InitialGrouping_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/DTSuperLayerId.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "DataFormats/DTDigi/interface/DTDigiCollection.h"

#include "L1Trigger/DTTriggerPhase2/interface/MuonPath.h"
#include "L1Trigger/DTTriggerPhase2/interface/constants.h"

#include "L1Trigger/DTTriggerPhase2/interface/MotherGrouping.h"

#include <iostream>
#include <fstream>

// ===============================================================================
// Previous definitions and declarations
// ===============================================================================

/*
  Channels are labeled following next schema:
    ---------------------------------
    |   6   |   7   |   8   |   9   |
    ---------------------------------
        |   3   |   4   |   5   |
        -------------------------
            |   1   |   2   |
            -----------------
                |   0   |
                ---------
*/

namespace dtamgrouping {
  /* Cell's combination, following previous labeling, to obtain every possible  muon's path. 
     Others cells combinations imply non straight paths */
  constexpr int CHANNELS_PATH_ARRANGEMENTS[8][4] = {
      {0, 1, 3, 6}, {0, 1, 3, 7}, {0, 1, 4, 7}, {0, 1, 4, 8}, {0, 2, 4, 7}, {0, 2, 4, 8}, {0, 2, 5, 8}, {0, 2, 5, 9}};

  /* For each of the previous cell's combinations, this array stores the associated cell's 
     displacement, relative to lower layer cell, measured in semi-cell length units */

  constexpr int CELL_HORIZONTAL_LAYOUTS[8][4] = {{0, -1, -2, -3},
                                                 {0, -1, -2, -1},
                                                 {0, -1, 0, -1},
                                                 {0, -1, 0, 1},
                                                 {0, 1, 0, -1},
                                                 {0, 1, 0, 1},
                                                 {0, 1, 2, 1},
                                                 {0, 1, 2, 3}};
}  // namespace dtamgrouping

// ===============================================================================
// Class declarations
// ===============================================================================

class InitialGrouping : public MotherGrouping {
public:
  // Constructors and destructor
  InitialGrouping(const edm::ParameterSet& pset, edm::ConsumesCollector& iC);
  ~InitialGrouping() override;

  // Main methods
  void initialise(const edm::EventSetup& iEventSetup) override;
  void run(edm::Event& iEvent,
           const edm::EventSetup& iEventSetup,
           const DTDigiCollection& digis,
           MuonPathPtrs& outMpath) override;
  void finish() override;

  // Other public methods

  // Public attributes

private:
  // Private methods
  void setInChannels(const DTDigiCollection* digi, int sl);
  void selectInChannels(int baseCh);
  void resetPrvTDCTStamp(void);
  void mixChannels(int sl, int pathId, MuonPathPtrs& outMpath);
  bool notEnoughDataInChannels(void);
  bool isEqualComb2Previous(DTPrimitives& ptr);

  // Private attributes
  const bool debug_;

  DTPrimitives muxInChannels_[cmsdt::NUM_CELLS_PER_BLOCK];
  DTPrimitives channelIn_[cmsdt::NUM_LAYERS][cmsdt::NUM_CH_PER_LAYER];
  DTPrimitives chInDummy_;
  int prevTDCTimeStamps_[4];
  int currentBaseChannel_;
};

#endif
