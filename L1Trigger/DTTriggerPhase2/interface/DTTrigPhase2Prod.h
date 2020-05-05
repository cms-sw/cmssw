#ifndef Phase2L1Trigger_DTTrigger_DTTrigPhase2Prod_cc
#define Phase2L1Trigger_DTTrigger_DTTrigPhase2Prod_cc
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"

#include "L1Trigger/DTTriggerPhase2/interface/MuonPath.h"
#include "L1Trigger/DTTriggerPhase2/interface/constants.h"

#include "L1Trigger/DTTriggerPhase2/interface/MotherGrouping.h"
#include "L1Trigger/DTTriggerPhase2/interface/InitialGrouping.h"
#include "L1Trigger/DTTriggerPhase2/interface/HoughGrouping.h"
#include "L1Trigger/DTTriggerPhase2/interface/PseudoBayesGrouping.h"
#include "L1Trigger/DTTriggerPhase2/interface/MuonPathAnalyzer.h"
#include "L1Trigger/DTTriggerPhase2/interface/MuonPathAnalyzerPerSL.h"
#include "L1Trigger/DTTriggerPhase2/interface/MuonPathAnalyzerInChamber.h"
#include "L1Trigger/DTTriggerPhase2/interface/MuonPathAssociator.h"
#include "L1Trigger/DTTriggerPhase2/interface/MPFilter.h"
#include "L1Trigger/DTTriggerPhase2/interface/MPQualityEnhancerFilter.h"
#include "L1Trigger/DTTriggerPhase2/interface/MPRedundantFilter.h"

#include "L1Trigger/DTSectorCollector/interface/DTSectCollPhSegm.h"
#include "L1Trigger/DTSectorCollector/interface/DTSectCollThSegm.h"

#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigManagerRcd.h"
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigManager.h"

#include "CalibMuon/DTDigiSync/interface/DTTTrigBaseSync.h"
#include "CalibMuon/DTDigiSync/interface/DTTTrigSyncFactory.h"

#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/DTSuperLayerId.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "DataFormats/DTDigi/interface/DTDigiCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment2DCollection.h"
#include "DataFormats/L1DTTrackFinder/interface/L1Phase2MuDTPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1Phase2MuDTPhDigi.h"

// DT trigger GeomUtils
#include "DQM/DTMonitorModule/interface/DTTrigGeomUtils.h"

//RPC TP
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"
#include <DataFormats/MuonDetId/interface/RPCDetId.h>
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "L1Trigger/DTTriggerPhase2/interface/RPCIntegrator.h"

#include <fstream>
#include <iostream>
#include <queue>
#include <cmath>

class DTTrigPhase2Prod : public edm::EDProducer {
  typedef std::map<DTChamberId, DTDigiCollection, std::less<DTChamberId>> DTDigiMap;
  typedef DTDigiMap::iterator DTDigiMap_iterator;
  typedef DTDigiMap::const_iterator DTDigiMap_const_iterator;

public:
  //! Constructor
  DTTrigPhase2Prod(const edm::ParameterSet& pset);

  //! Destructor
  ~DTTrigPhase2Prod() override;

  //! Create Trigger Units before starting event processing
  //void beginJob(const edm::EventSetup & iEventSetup);
  void beginRun(edm::Run const& iRun, const edm::EventSetup& iEventSetup) override;

  //! Producer: process every event and generates trigger data
  void produce(edm::Event& iEvent, const edm::EventSetup& iEventSetup) override;

  //! endRun: finish things
  void endRun(edm::Run const& iRun, const edm::EventSetup& iEventSetup) override;

  // Methods
  int rango(metaPrimitive mp);
  bool outer(metaPrimitive mp);
  bool inner(metaPrimitive mp);
  void printmP(metaPrimitive mP);
  void printmPC(metaPrimitive mP);
  double trigPos(metaPrimitive mP);
  double trigDir(metaPrimitive mp);
  bool hasPosRF(int wh, int sec);

  // Getter-methods
  MP_QUALITY getMinimumQuality(void);

  // Setter-methods
  void setChiSquareThreshold(float ch2Thr);
  void setMinimumQuality(MP_QUALITY q);

  // data-members
  edm::ESHandle<DTGeometry> dtGeo_;
  std::vector<std::pair<int, MuonPath>> primitives_;

private:
  // Trigger Configuration Manager CCB validity flag
  bool my_CCBValid_;

  // BX offset used to correct DTTPG output
  int my_BXoffset_;

  // Debug Flag
  bool debug_;
  bool dump_;
  double dT0_correlate_TP_;
  bool do_correlation_;
  int scenario_;

  // shift
  edm::FileInPath shift_filename_;
  std::map<int, float> shiftinfo_;

  // ParameterSet
  edm::EDGetTokenT<DTRecSegment4DCollection> dt4DSegmentsToken_;
  edm::EDGetTokenT<DTDigiCollection> dtDigisToken_;
  edm::EDGetTokenT<RPCRecHitCollection> rpcRecHitsLabel_;

  // Grouping attributes and methods
  int grcode_;  // Grouping code
  MotherGrouping* grouping_obj_;
  MuonPathAnalyzer* mpathanalyzer_;
  MPFilter* mpathqualityenhancer_;
  MPFilter* mpathredundantfilter_;
  MuonPathAssociator* mpathassociator_;

  // Buffering
  bool activateBuffer_;
  int superCellhalfspacewidth_;
  float superCelltimewidth_;
  std::vector<DTDigiCollection*> distribDigis(std::queue<std::pair<DTLayerId*, DTDigi*>>& inQ);
  void processDigi(std::queue<std::pair<DTLayerId*, DTDigi*>>& inQ,
                   std::vector<std::queue<std::pair<DTLayerId*, DTDigi*>>*>& vec);

  // RPC
  RPCIntegrator* rpc_integrator_;
  bool useRPC_;

  void assignIndex(std::vector<metaPrimitive>& inMPaths);
  void assignIndexPerBX(std::vector<metaPrimitive>& inMPaths);
  int assignQualityOrder(metaPrimitive mP);
};

#endif
