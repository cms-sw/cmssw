#ifndef L1Trigger_DTTriggerPhase2_HoughGrouping_h
#define L1Trigger_DTTriggerPhase2_HoughGrouping_h

// System / std headers
#include <memory>
#include <cstdlib>
#include <stdexcept>
#include <iostream>
#include <vector>

// Other headers
#include "L1Trigger/DTTriggerPhase2/interface/MuonPath.h"
#include "L1Trigger/DTTriggerPhase2/interface/DTprimitive.h"
#include "L1Trigger/DTTriggerPhase2/interface/MotherGrouping.h"

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

typedef std::pair<double, double> PointInPlane;
typedef std::vector<PointInPlane> PointsInPlane;
typedef std::tuple<double, double, unsigned short int> PointTuple;
typedef std::vector<PointTuple> PointTuples;
typedef std::map<unsigned short int, double> PointMap;

class HoughGrouping : public MotherGrouping {
public:
  // Constructors and destructor
  HoughGrouping(const edm::ParameterSet& pset, edm::ConsumesCollector& iC);
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

  PointsInPlane getMaximaVector();
  PointsInPlane findTheMaxima(PointTuples& inputvec);

  PointInPlane getTwoDelta(const PointTuple& pair1, const PointTuple& pair2);
  PointInPlane getAveragePoint(const PointTuples& inputvec,
                               unsigned short int firstindex,
                               const std::vector<unsigned short int>& indexlist);
  PointInPlane transformPair(const PointInPlane& inputpair);

  ProtoCand associateHits(const DTChamber* thechamb, double m, double n);

  void orderAndFilter(std::vector<ProtoCand>& invector, MuonPathPtrs& outMuonPath);

  void setDifferenceBetweenSL(ProtoCand& tupl);
  bool areThereEnoughHits(const ProtoCand& tupl);

  // Private attributes
  const bool debug_;
  bool allowUncorrelatedPatterns_;
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

  std::vector<std::vector<unsigned short int>> linespace_;

  PointMap anglemap_;
  PointMap posmap_;
  std::map<unsigned short int, DTPrimitive> digimap_[8];

  PointsInPlane maxima_;
  PointsInPlane hitvec_;
};

#endif
