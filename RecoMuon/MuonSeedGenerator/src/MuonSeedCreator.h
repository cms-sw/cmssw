#ifndef RecoMuon_SeedGenerator_MuonSeedCreator_H
#define RecoMuon_SeedGenerator_MuonSeedCreator_H

/** \class MuonSeedCreator
 *  
 * Creates seed from vector of segment
 *
 * Determine pt of seed using various combination of segments from different layers (stations)
 * Parameterization used to determine pt between layers i and j:
 *
 * pt = [ c_0 + c_1 * (Delta phi_ij) + c_2 * (Delta phi_ij)^2 ] / eta
 *
 * \author Dominique Fortin - UCR 
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
  TrajectorySeed createSeed(int type, SegmentContainer seg, std::vector<int> layers, std::vector<int> badSeedLayer);

  
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

  /// Compute pt from parameters
  std::vector<double> getPt(std::vector<double> vParameters, double eta, double dPhi);

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

  // seed parameters vectors
  std::vector<double> DT12;  
  std::vector<double> DT13;  
  std::vector<double> DT14;  
  std::vector<double> DT23;  
  std::vector<double> DT24;  
  std::vector<double> DT34;  

  std::vector<double> CSC01;
  std::vector<double> CSC12;
  std::vector<double> CSC02;
  std::vector<double> CSC13;
  std::vector<double> CSC03;
  std::vector<double> CSC14;
  std::vector<double> CSC23;
  std::vector<double> CSC24;
  std::vector<double> CSC34;

  std::vector<double> OL1213;
  std::vector<double> OL1222;
  std::vector<double> OL1232;
  std::vector<double> OL2213;
  std::vector<double> OL2222;
  
  std::vector<double> SME11;
  std::vector<double> SME12;
  std::vector<double> SME13;
  std::vector<double> SME21;
  std::vector<double> SME22;
  std::vector<double> SME31;
  std::vector<double> SME32;
  std::vector<double> SME41;

  std::vector<double> SMB10;
  std::vector<double> SMB11;
  std::vector<double> SMB12;
  std::vector<double> SMB20;
  std::vector<double> SMB21;
  std::vector<double> SMB22;
  std::vector<double> SMB30;
  std::vector<double> SMB31;
  std::vector<double> SMB32;
};
#endif
