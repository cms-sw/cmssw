#ifndef RecoMuon_SeedGenerator_MuonSeedCreator_H
#define RecoMuon_SeedGenerator_MuonSeedCreator_H

/** \class MuonSeedCreator
 *  
 * Creates seed from vector of segment
 *
 * \author Dominique Fortin  
 */

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <vector>

class RecHit;
class BoundPlane;
class GeomDet;
class MagneticField;
class MuonTransientTrackingRecHit;

class MuonSeedCreator {

 public:

  typedef MuonTransientTrackingRecHit::MuonRecHitContainer SegmentContainer;

  /// Constructor
  explicit MuonSeedCreator(const edm::ParameterSet& pset);

  /// Destructor
  ~MuonSeedCreator();

  // Operations

  /// Cache Magnetic Field for current event
  void setBField( const MagneticField* theField ){ BField = theField; };

  /// Create a seed from set of segments
  TrajectorySeed createSeed(int type, SegmentContainer seg, std::vector<int> layers, std::vector<int> badSeedLayer, float& ptSeed);

  
 private:

  /// Estimate transverse momentum of track from CSC measurements
  void estimatePtCSC(SegmentContainer seg, std::vector<int> layers, double& pt, double& spt);

  /// Estimate transverse momentum of track from CSC + DT measurements
  void estimatePtOverlap(SegmentContainer seg, std::vector<int> layers, double& pt, double& spt);

  /// Estimate transverse momentum of track from DT measurements
  void estimatePtDT(SegmentContainer seg, std::vector<int> layers, double& pt, double& spt);

  /// Estimate transverse momentum of track from single segment
  void estimatePtSingle(SegmentContainer seg, std::vector<int> layers, double& pt, double& spt);

  /// Compute weighted mean pt from different pt estimators
  void weightedPt(std::vector<double> ptEstimate, std::vector<double> sptEstimate, double& ptAvg, double& sptAvg);

  // Miminum and maximum pt momentum of a track  
  float theMinMomentum;
  float theMaxMomentum;
  float defaultMomentum;

  // Error on pt estimate which prevents weighted average from blowing up ( spt --> 0 )
  double sysError;
 
  // Flag for internal debugging
  bool debug;
 
  // Cache Magnetic Field for current event
  const MagneticField* BField;

};
#endif
