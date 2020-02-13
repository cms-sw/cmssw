#ifndef L1Trigger_DTPhase2Trigger_HoughGrouping_cc
#define L1Trigger_DTPhase2Trigger_HoughGrouping_cc

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
#include "DataFormats/MuonDetId/interface/DTLayerId.h" // New trying to avoid crashes in the topology functions
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

#include "SimDataFormats/DigiSimLinks/interface/DTDigiSimLinkCollection.h"// To cope with Digis from simulation
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
#include "Geometry/DTGeometry/interface/DTLayer.h" // New trying to avoid crashes in the topology functions
#include "Geometry/DTGeometry/interface/DTTopology.h" // New trying to avoid crashes in the topology functions

// ROOT headers
#include "TROOT.h"
#include "TMath.h"

// Other headers
#include "L1Trigger/DTPhase2Trigger/interface/muonpath.h"
#include "L1Trigger/DTPhase2Trigger/interface/dtprimitive.h"
#include "L1Trigger/DTPhase2Trigger/interface/MotherGrouping.h"

// ===============================================================================
// Previous definitions and declarations
// ===============================================================================
// Namespaces
using namespace std;
using namespace edm;
using namespace cms;


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
    void run(edm::Event& iEvent, const edm::EventSetup& iEventSetup, DTDigiCollection digis, std::vector<MuonPath*> *outMpath) override;
    void finish() override;
    
    // Other public methods
    
    // Public attributes
    
  private:
    // Private methods
    void ResetAttributes();
    void ResetPosElementsOfLinespace();
    
    void ObtainGeometricalBorders(const DTLayer* lay);
    
    void DoHoughTransform();
    
    std::vector<std::pair<Double_t, Double_t>> GetMaximaVector();
    std::vector<std::pair<Double_t, Double_t>> FindTheMaxima(std::vector<std::tuple<Double_t, Double_t, UShort_t>> inputvec);
    
    std::pair<Double_t, Double_t> GetTwoDelta(std::tuple<Double_t, Double_t, UShort_t> pair1, std::tuple<Double_t, Double_t, UShort_t> pair2);
    std::pair<Double_t, Double_t> GetAveragePoint(std::vector<std::tuple<Double_t, Double_t, UShort_t>> inputvec, UShort_t firstindex, std::vector<UShort_t> indexlist);
    std::pair<Double_t, Double_t> TransformPair(std::pair<Double_t, Double_t> inputpair);
    
    std::tuple<UShort_t, Bool_t*, Bool_t*, UShort_t, Double_t*, DTPrimitive*> AssociateHits(const DTChamber* thechamb, Double_t m, Double_t n);
    
    void OrderAndFilter(std::vector<std::tuple<UShort_t, Bool_t*, Bool_t*, UShort_t, Double_t*, DTPrimitive*>> &invector, std::vector<MuonPath*> *&outMuonPath);
    
    void   SetDifferenceBetweenSL(std::tuple<UShort_t, Bool_t*, Bool_t*, UShort_t, Double_t*, DTPrimitive*> &tupl);
    Bool_t AreThereEnoughHits(std::tuple<UShort_t, Bool_t*, Bool_t*, UShort_t, Double_t*, DTPrimitive*> tupl);
    
    // Private attributes
    Bool_t   debug, allowUncorrelatedPatterns;
    UShort_t minNLayerHits, minSingleSLHitsMax, minSingleSLHitsMin, minUncorrelatedHits, UpperNumber, LowerNumber;
    Double_t angletan, anglebinwidth, posbinwidth, maxdeltaAngDeg, maxdeltaPos, MaxDistanceToWire;
    
    edm::ESHandle<DTGeometry> dtGeomH;
    DTChamberId TheChambId;
    
    Double_t maxrads, minangle, oneanglebin;
    Double_t xlowlim, xhighlim, zlowlim, zhighlim;
    Double_t maxdeltaAng;
    
    UShort_t anglebins, halfanglebins, spacebins;
    UShort_t idigi, nhits;
    UShort_t thestation, thesector;
    Short_t  thewheel;
    
    UShort_t** linespace;
    
    std::map<UShort_t, Double_t> anglemap;
    std::map<UShort_t, Double_t> posmap;
    std::map<UShort_t, DTPrimitive> digimap [8];
    
    std::vector<std::pair<Double_t, Double_t>> maxima;
    std::vector<std::pair<Double_t, Double_t>> hitvec;
};

#endif
