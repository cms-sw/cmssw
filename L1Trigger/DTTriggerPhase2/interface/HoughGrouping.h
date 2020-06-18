#ifndef L1Trigger_DTTriggerPhase2_HoughGrouping_h
#define L1Trigger_DTTriggerPhase2_HoughGrouping_h

// System / std headers
#include <memory>
#include <cstdlib>
#include <stdexcept>
#include <iostream>
#include <vector>

// CMSSW headers
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/FrameworkfwdMostUsed.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/DTGeometry/interface/DTTopology.h"

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
struct ProtoCand {
  unsigned short int nLayersWithHits_;   // 0: # of layers with hits.
  std::vector<bool> isThereHitInLayer_;  // 1: # of hits of high quality (the expected line crosses the cell).
  std::vector<bool>
      isThereNeighBourHitInLayer_;      // 2: # of hits of low quality (the expected line is in a neighbouring cell).
  unsigned short int nHitsDiff_;        // 3: absolute diff. between the number of hits in SL1 and SL3.
  std::vector<double> xDistToPattern_;  // 4: absolute distance to all hits of the segment.
  DTPrimitives dtHits_;                 // 5: DTPrimitive of the candidate.
};

class HoughGrouping : public MotherGrouping {
public:
  // Constructors and destructor
  HoughGrouping(const ParameterSet& pset, edm::ConsumesCollector& iC);
  ~HoughGrouping() override;

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
  void resetAttributes();
  void resetPosElementsOfLinespace();

  void obtainGeometricalBorders(const DTLayer* lay);

  void doHoughTransform();

  std::vector<std::pair<double, double>> getMaximaVector();
  std::vector<std::pair<double, double>> findTheMaxima(
      std::vector<std::tuple<double, double, unsigned short int>> inputvec);

  std::pair<double, double> getTwoDelta(std::tuple<double, double, unsigned short int> pair1,
                                        std::tuple<double, double, unsigned short int> pair2);
  std::pair<double, double> getAveragePoint(std::vector<std::tuple<double, double, unsigned short int>> inputvec,
                                            unsigned short int firstindex,
                                            std::vector<unsigned short int> indexlist);
  std::pair<double, double> transformPair(std::pair<double, double> inputpair);

  ProtoCand associateHits(const DTChamber* thechamb, double m, double n);

  void orderAndFilter(std::vector<ProtoCand>& invector, MuonPathPtrs& outMuonPath);

  void setDifferenceBetweenSL(ProtoCand& tupl);
  bool areThereEnoughHits(ProtoCand& tupl);

  // Private attributes
  bool debug_, allowUncorrelatedPatterns_;
  unsigned short int minNLayerHits_, minSingleSLHitsMax_, minSingleSLHitsMin_, minUncorrelatedHits_, upperNumber_,
      lowerNumber_;
  double angletan_, anglebinwidth_, posbinwidth_, maxdeltaAngDeg_, maxdeltaPos_, maxDistanceToWire_;

  DTGeometry const* dtGeo_;
  edm::ESGetToken<DTGeometry, MuonGeometryRecord> dtGeomH;
  DTChamberId TheChambId;

  double maxrads_, minangle_, oneanglebin_;
  double xlowlim_, xhighlim_, zlowlim_, zhighlim_;
  double maxdeltaAng_;

  unsigned short int anglebins_, halfanglebins_, spacebins_;
  unsigned short int idigi_, nhits_;
  unsigned short int thestation_, thesector_;
  short int thewheel_;

  unsigned short int** linespace_;

  std::map<unsigned short int, double> anglemap_;
  std::map<unsigned short int, double> posmap_;
  std::map<unsigned short int, DTPrimitive> digimap_[8];

  std::vector<std::pair<double, double>> maxima_;
  std::vector<std::pair<double, double>> hitvec_;
};

#endif
