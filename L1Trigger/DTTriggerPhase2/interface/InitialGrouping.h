#ifndef Phase2L1Trigger_DTTrigger_InitialGrouping_cc
#define Phase2L1Trigger_DTTrigger_InitialGrouping_cc

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/DTSuperLayerId.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "DataFormats/DTDigi/interface/DTDigiCollection.h"

#include "L1Trigger/DTTriggerPhase2/interface/MuonPath.h"
#include "L1Trigger/DTTriggerPhase2/interface/constants.h"

#include "L1Trigger/DTTriggerPhase2/interface/MotherGrouping.h"

#include "CalibMuon/DTDigiSync/interface/DTTTrigBaseSync.h"
#include "CalibMuon/DTDigiSync/interface/DTTTrigSyncFactory.h"

#include "L1Trigger/DTSectorCollector/interface/DTSectCollPhSegm.h"
#include "L1Trigger/DTSectorCollector/interface/DTSectCollThSegm.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"

#include <iostream>
#include <fstream>

#define MAX_VERT_ARRANG 4

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

/* Cell's combination, following previous labeling, to obtain every possible  muon's path. Others cells combinations imply non straight paths */
constexpr int CHANNELS_PATH_ARRANGEMENTS[8][4] = {
    {0, 1, 3, 6}, {0, 1, 3, 7}, {0, 1, 4, 7}, {0, 1, 4, 8}, {0, 2, 4, 7}, {0, 2, 4, 8}, {0, 2, 5, 8}, {0, 2, 5, 9}};

/* For each of the previous cell's combinations, this array stores the associated cell's displacement, relative to lower layer cell, measured in semi-cell length units */

constexpr int CELL_HORIZONTAL_LAYOUTS[8][4] = {{0, -1, -2, -3},
                                               {0, -1, -2, -1},
                                               {0, -1, 0, -1},
                                               {0, -1, 0, 1},
                                               {0, 1, 0, -1},
                                               {0, 1, 0, 1},
                                               {0, 1, 2, 1},
                                               {0, 1, 2, 3}};

// ===============================================================================
// Class declarations
// ===============================================================================

class InitialGrouping : public MotherGrouping {
  typedef std::map<DTChamberId, DTDigiCollection, std::less<DTChamberId> > DTDigiMap;
  typedef DTDigiMap::iterator DTDigiMap_iterator;
  typedef DTDigiMap::const_iterator DTDigiMap_const_iterator;

public:
  // Constructors and destructor
  InitialGrouping(const edm::ParameterSet& pset, edm::ConsumesCollector& iC);
  ~InitialGrouping() override;

  // Main methods
  void initialise(const edm::EventSetup& iEventSetup) override;
  void run(edm::Event& iEvent,
           const edm::EventSetup& iEventSetup,
           const DTDigiCollection& digis,
           std::vector<MuonPath*>* outMpath) override;
  void finish() override;

  // Other public methods

  // Public attributes

private:
  // Private methods
  void setInChannels(const DTDigiCollection* digi, int sl);
  void selectInChannels(int baseCh);
  void resetPrvTDCTStamp(void);
  void mixChannels(int sl, int pathId, std::vector<MuonPath*>* outMpath);
  bool notEnoughDataInChannels(void);
  bool isEqualComb2Previous(DTPrimitive* ptr[4]);

  // Private attributes
  bool debug;
  std::string ttrig_filename;
  std::map<int, float> ttriginfo;

  std::vector<DTPrimitive> muxInChannels[NUM_CELLS_PER_BLOCK];
  std::vector<DTPrimitive> channelIn[NUM_LAYERS][NUM_CH_PER_LAYER];
  std::vector<DTPrimitive> chInDummy;
  int prevTDCTimeStamps[4];
  int currentBaseChannel;
};

#endif
