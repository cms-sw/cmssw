#ifndef Phase2L1Trigger_DTTrigger_TrapezoidalGrouping_h
#define Phase2L1Trigger_DTTrigger_TrapezoidalGrouping_h

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
#include <stack>

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

inline bool hitWireSort(const DTPrimitive& hit1, const DTPrimitive& hit2) {
  int wi1 = hit1.channelId();
  int wi2 = hit2.channelId();

  if (wi1 < wi2)
    return true;
  else
    return false;
}

inline bool hitLayerSort(const DTPrimitive& hit1, const DTPrimitive& hit2) {
  int lay1 = hit1.layerId();
  int lay2 = hit2.layerId();

  if (lay1 < lay2)
    return true;
  else if (lay1 > lay2)
    return false;
  else
    return hitWireSort(hit1, hit2);
}

inline bool hitTimeSort(const DTPrimitive& hit1, const DTPrimitive& hit2) {
  int tdc1 = hit1.tdcTimeStamp();
  int tdc2 = hit2.tdcTimeStamp();

  if (tdc1 < tdc2)
    return true;
  else if (tdc1 > tdc2)
    return false;
  else
    return hitLayerSort(hit1, hit2);
}

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

class TrapezoidalGrouping : public MotherGrouping {
public:
  // Constructors and destructor
  TrapezoidalGrouping(const edm::ParameterSet& pset, edm::ConsumesCollector& iC);
  ~TrapezoidalGrouping() override;

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
  std::vector<DTPrimitives> group_hits(DTPrimitive pivot_hit,
                                       std::vector<DTPrimitives> input_paths,
                                       DTPrimitives hits_per_cell,
                                       DTPrimitives& hits_in_trapezoid);

  // Private attributes
  const bool debug_;

  DTPrimitives muxInChannels_[cmsdt::NUM_CELLS_PER_BLOCK];
  DTPrimitives channelIn_[cmsdt::NUM_LAYERS][cmsdt::NUM_CH_PER_LAYER];
  DTPrimitives all_hits;
  DTPrimitives chInDummy_;
  int prevTDCTimeStamps_[4];
  int currentBaseChannel_;

  // The trapezoid is as follows:
  // [  0  ][  1  ][  2  ][  3  ][  4  ][  5  ][  6  ][  7  ][  8  ]

  // And maps to the physical cells as follows:

  // Pivot in layer 1 = "00"
  // [  5  ][  6  ][  7  ][  8  ] Layer C
  //    [  2  ][  3  ][  4  ]     Layer B
  //        [  0  ][  1  ]        Layer A
  //            Pivot

  // Pivot in layer 2 = "01"
  //    [  2  ][  3  ][  4  ]     Layer B
  //        [  0  ][  1  ]        Layer A
  //            Pivot
  //        [ 6,8 ][ 5,7 ]        Layer C

  // Pivot in layer 3 = "10"
  //        [ 6,8 ][ 5,7 ]        Layer C
  //            Pivot
  //        [  0  ][  1  ]        Layer A
  //    [  2  ][  3  ][  4  ]     Layer B

  // Pivot in layer 4 = "11"
  //            Pivot
  //        [  0  ][  1  ]        Layer A
  //    [  2  ][  3  ][  4  ]     Layer B
  // [  5  ][  6  ][  7  ][  8  ] Layer C

  short trapezoid_vertical_mapping[4][9] = {{1, 1, 2, 2, 2, 3, 3, 3, 3},
                                            {1, 1, 2, 2, 2, -1, -1, -1, -1},
                                            {-1, -1, -2, -2, -2, 1, 1, 1, 1},
                                            {-1, -1, -2, -2, -2, -3, -3, -3, -3}};

  short trapezoid_horizontal_mapping[4][9] = {{0, 1, -1, 0, 1, -1, 0, 1, 2},
                                              {-1, 0, -1, 0, 1, 0, -1, 0, -1},
                                              {0, 1, -1, 0, 1, 1, 0, 1, 0},
                                              {-1, 0, -1, 0, 1, -2, -1, 0, 1}};

  // Task list
  // 4 hit candidates
  // 0 => (0,2,5),       1 => (0,2,6),       2 => (0,3,6),       3 => (0,3,7),
  // 4 => (1,3,6),       5 => (1,3,7),       6 => (1,4,7),       7 => (1,4,8),
  // the rest are 3-hit candidates, last value not used
  // 8 => (0,2,0),       9 => (0,3,0),      10 => (1,3,0),      11 => (1,4,0),
  // 12 => (0,5,0),      13 => (0,6,0),      14 => (0,7,0),      15 => (1,6,0),
  // 16 => (1,7,0),      17 => (1,8,0),      18 => (2,5,0),      19 => (2,6,0),
  // 20 => (3,6,0),      21 => (3,7,0),      22 => (4,7,0),      23 => (4,8,0)

  std::vector<std::vector<short>> task_list = {// 4-hit
                                               {0, 2, 5},
                                               {0, 2, 6},
                                               {0, 3, 6},
                                               {0, 3, 7},
                                               {1, 3, 6},
                                               {1, 3, 7},
                                               {1, 4, 7},
                                               {1, 4, 8},
                                               // 3-hit
                                               {0, 2},
                                               {0, 3},
                                               {1, 3},
                                               {1, 4},
                                               {0, 5},
                                               {0, 6},
                                               {0, 7},
                                               {1, 6},
                                               {1, 7},
                                               {1, 8},
                                               {2, 5},
                                               {2, 6},
                                               {3, 6},
                                               {3, 7},
                                               {4, 7},
                                               {4, 8}};

  int CELL_HORIZONTAL_LAYOUTS_PER_TASK[4][24][4] = { // pivoting over layer 1
                                                    {// all layers available
                                                     {0, 0, 0, -1},
                                                     {0, 0, 1, -1},
                                                     {0, 1, 0, -1},
                                                     {0, 1, 1, -1},
                                                     {1, 0, 0, -1},
                                                     {1, 0, 1, -1},
                                                     {1, 1, 0, -1},
                                                     {1, 1, 1, -1},
                                                     // layer 4 missing
                                                     {0, 0, 0, -1},
                                                     {0, 1, 0, -1},
                                                     {1, 0, 0, -1},
                                                     {1, 1, 0, -1},
                                                     // layer 3 missing
                                                     {0, 0, 0, -1},
                                                     {0, 0, 1, -1},
                                                     {0, 1, 1, -1},
                                                     {1, 0, 0, -1},
                                                     {1, 0, 1, -1},
                                                     {1, 1, 1, -1},
                                                     // layer 2 missing
                                                     {0, 0, 0, -1},
                                                     {0, 0, 1, -1},
                                                     {0, 1, 0, -1},
                                                     {0, 1, 1, -1},
                                                     {1, 1, 0, -1},
                                                     {1, 1, 1, -1}},
                                                    // pivoting over layer 2
                                                    {// all layers available
                                                     {0, 0, 0, -1},
                                                     {1, 0, 0, -1},
                                                     {1, 0, 1, -1},
                                                     {0, 0, 1, -1},
                                                     {1, 1, 0, -1},
                                                     {0, 1, 0, -1},
                                                     {0, 1, 1, -1},
                                                     {1, 1, 1, -1},
                                                     // layer 1 missing
                                                     {0, 0, 0, -1},
                                                     {0, 0, 1, -1},
                                                     {0, 1, 0, -1},
                                                     {0, 1, 1, -1},
                                                     // layer 4 missing
                                                     {0, 0, 0, -1},
                                                     {1, 0, 0, -1},
                                                     {0, 0, 0, -1},
                                                     {1, 1, 0, -1},
                                                     {0, 1, 0, -1},
                                                     {1, 1, 0, -1},
                                                     // layer 3 missing
                                                     {0, 0, 0, -1},
                                                     {1, 0, 0, -1},
                                                     {1, 0, 1, -1},
                                                     {0, 0, 1, -1},
                                                     {0, 1, 1, -1},
                                                     {1, 1, 1, -1}},
                                                    // pivoting over layer 3
                                                    {// all layers available
                                                     {1, 1, 1, -1},
                                                     {1, 1, 0, -1},
                                                     {0, 1, 0, -1},
                                                     {0, 1, 1, -1},
                                                     {1, 0, 0, -1},
                                                     {1, 0, 1, -1},
                                                     {0, 0, 1, -1},
                                                     {0, 0, 0, -1},
                                                     // layer 4 missing
                                                     {1, 1, 0, -1},
                                                     {0, 1, 0, -1},
                                                     {1, 0, 0, -1},
                                                     {0, 0, 0, -1},
                                                     // layer 1 missing
                                                     {0, 1, 1, -1},
                                                     {0, 1, 0, -1},
                                                     {0, 1, 1, -1},
                                                     {0, 0, 0, -1},
                                                     {0, 0, 1, -1},
                                                     {0, 0, 0, -1},
                                                     // layer 2 missing
                                                     {1, 1, 1, -1},
                                                     {1, 1, 0, -1},
                                                     {0, 1, 0, -1},
                                                     {0, 1, 1, -1},
                                                     {0, 0, 1, -1},
                                                     {0, 0, 0, -1}},
                                                    // pivoting over layer 4
                                                    {// all layers available
                                                     {1, 1, 1, -1},
                                                     {0, 1, 1, -1},
                                                     {1, 0, 1, -1},
                                                     {0, 0, 1, -1},
                                                     {1, 1, 0, -1},
                                                     {0, 1, 0, -1},
                                                     {1, 0, 0, -1},
                                                     {0, 0, 0, -1},
                                                     // layer 1 missing
                                                     {0, 1, 1, -1},
                                                     {0, 0, 1, -1},
                                                     {0, 1, 0, -1},
                                                     {0, 0, 0, -1},
                                                     // layer 2 missing
                                                     {1, 1, 1, -1},
                                                     {0, 1, 1, -1},
                                                     {0, 0, 1, -1},
                                                     {1, 1, 0, -1},
                                                     {0, 1, 0, -1},
                                                     {0, 0, 0, -1},
                                                     // layer 3 missing
                                                     {1, 1, 1, -1},
                                                     {0, 1, 1, -1},
                                                     {1, 0, 1, -1},
                                                     {0, 0, 1, -1},
                                                     {1, 0, 0, -1},
                                                     {0, 0, 0, -1}}};

  int MISSING_LAYER_LAYOUTS_PER_TASK[4][24] = {
      {-1, -1, -1, -1, -1, -1, -1, -1, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1},
      {-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2},
      {-1, -1, -1, -1, -1, -1, -1, -1, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1},
      {-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2}};
};

#endif
