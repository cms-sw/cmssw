#ifndef L1Trigger_DTTriggerPhase2_HoughGrouping_cc
#define L1Trigger_DTTriggerPhase2_HoughGrouping_cc

// System / std headers
#include <memory>
#include <cstdlib>
#include <stdexcept>
#include <iostream>
#include <vector>

// CMSSW headers
#include <FWCore/Framework/interface/ConsumesCollector.h>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Common/interface/TriggerNames.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "DataFormats/MuonData/interface/MuonDigiCollection.h"

#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCandFwd.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
#include "DataFormats/L1Trigger/interface/Muon.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/CSCRecHit/interface/CSCSegmentCollection.h"
#include "DataFormats/DTDigi/interface/DTDigi.h"
#include "DataFormats/DTDigi/interface/DTDigiCollection.h"
#include "DataFormats/DTDigi/interface/DTLocalTriggerCollection.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "DataFormats/RPCDigi/interface/RPCDigi.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"  // New trying to avoid crashes in the topology functions
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/Scalers/interface/LumiScalers.h"

#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/GeomPropagators/interface/StraightLinePlaneCrossing.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

#include "MagneticField/Engine/interface/MagneticField.h"

#include "SimDataFormats/DigiSimLinks/interface/DTDigiSimLinkCollection.h"  // To cope with Digis from simulation
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"

#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"
#include "CondFormats/DTObjects/interface/DTMtime.h"
#include "CondFormats/DataRecord/interface/DTMtimeRcd.h"

#include "CalibMuon/DTDigiSync/interface/DTTTrigBaseSync.h"
#include "CalibMuon/DTDigiSync/interface/DTTTrigSyncFactory.h"
// #include "CalibMuon/DTCalibration/plugins/DTCalibrationMap.h"

#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"  // New trying to avoid crashes in the topology functions
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"     // New trying to avoid crashes in the topology functions
#include "Geometry/DTGeometry/interface/DTTopology.h"  // New trying to avoid crashes in the topology functions

// ROOT headers
#include "TROOT.h"
#include "TMath.h"

// Other headers
#include "L1Trigger/DTTriggerPhase2/interface/MuonPath.h"
#include "L1Trigger/DTTriggerPhase2/interface/DTprimitive.h"
#include "L1Trigger/DTTriggerPhase2/interface/MotherGrouping.h"

// ===============================================================================
// Previous definitions and declarations
// ===============================================================================
// Namespaces
using namespace edm;
using namespace cmsdt;

// ===============================================================================
// Class declarations
// ===============================================================================
class HoughGrouping : public MotherGrouping {
public:
  // Constructors and destructor
  HoughGrouping(const ParameterSet& pset);
  ~HoughGrouping() override;

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
  void ResetAttributes();
  void ResetPosElementsOfLinespace();

  void ObtainGeometricalBorders(const DTLayer* lay);

  void DoHoughTransform();

  std::vector<std::pair<double, double>> GetMaximaVector();
  std::vector<std::pair<double, double>> FindTheMaxima(
      std::vector<std::tuple<double, double, unsigned short int>> inputvec);

  std::pair<double, double> GetTwoDelta(std::tuple<double, double, unsigned short int> pair1,
                                            std::tuple<double, double, unsigned short int> pair2);
  std::pair<double, double> GetAveragePoint(std::vector<std::tuple<double, double, unsigned short int>> inputvec,
                                                unsigned short int firstindex,
                                                std::vector<unsigned short int> indexlist);
  std::pair<double, double> TransformPair(std::pair<double, double> inputpair);

  std::tuple<unsigned short int, bool*, bool*, unsigned short int, double*, DTPrimitive*> AssociateHits(const DTChamber* thechamb,
                                                                                          double m,
                                                                                          double n);

  void OrderAndFilter(std::vector<std::tuple<unsigned short int, bool*, bool*, unsigned short int, double*, DTPrimitive*>>& invector,
                      std::vector<MuonPath*>*& outMuonPath);

  void SetDifferenceBetweenSL(std::tuple<unsigned short int, bool*, bool*, unsigned short int, double*, DTPrimitive*>& tupl);
  bool AreThereEnoughHits(std::tuple<unsigned short int, bool*, bool*, unsigned short int, double*, DTPrimitive*> tupl);

  // Private attributes
  bool debug, allowUncorrelatedPatterns;
  unsigned short int minNLayerHits, minSingleSLHitsMax, minSingleSLHitsMin, minUncorrelatedHits, UpperNumber, LowerNumber;
  double angletan, anglebinwidth, posbinwidth, maxdeltaAngDeg, maxdeltaPos, MaxDistanceToWire;

  edm::ESGetToken<DTGeometry, MuonGeometryRecord> dtGeomH;
  DTChamberId TheChambId;

  double maxrads, minangle, oneanglebin;
  double xlowlim, xhighlim, zlowlim, zhighlim;
  double maxdeltaAng;

  unsigned short int anglebins, halfanglebins, spacebins;
  unsigned short int idigi, nhits;
  unsigned short int thestation, thesector;
  short int thewheel;

  unsigned short int** linespace;

  std::map<unsigned short int, double> anglemap;
  std::map<unsigned short int, double> posmap;
  std::map<unsigned short int, DTPrimitive> digimap[8];

  std::vector<std::pair<double, double>> maxima;
  std::vector<std::pair<double, double>> hitvec;
};

#endif
