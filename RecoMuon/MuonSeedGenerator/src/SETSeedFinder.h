#ifndef MuonSeedGenerator_SETSeedFinder_h
#define MuonSeedGenerator_SETSeedFinder_h

#include "RecoMuon/MuonSeedGenerator/src/MuonSeedVFinder.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "RecoMuon/MuonSeedGenerator/src/MuonSeedPtExtractor.h"
#include "RecoMuon/MuonSeedGenerator/src/SETFilter.h"
#include "CLHEP/Matrix/Vector.h"
#include "CLHEP/Vector/ThreeVector.h"


class SETSeedFinder : public MuonSeedVFinder 
{
public:
  typedef MuonTransientTrackingRecHit::MuonRecHitContainer MuonRecHitContainer;

  explicit SETSeedFinder(const edm::ParameterSet & pset);
  virtual ~SETSeedFinder() {delete thePtExtractor;}
  /// ignore - uses MuonServiceProxy
  virtual void setBField(const MagneticField * field) {}

  /** The container sent in is expected to be a cluster, which isn't the same as
      a pattern.  A cluster can have more than one hit on a layer.  Internally,
      this method splits the cluster into patterns, and chooses the best one via a chi2.
      But it calculates the trajectoryMeasurements at the same time, so we can't
      really separate the steps.
  */

  virtual void seeds(const MuonRecHitContainer & cluster,
                     std::vector<TrajectorySeed> & result);

  void setServiceProxy(MuonServiceProxy * service) {theService = service;}

  std::vector<MuonRecHitContainer>
  sortByLayer(MuonRecHitContainer & cluster) const;

  //---- For protection against huge memory consumtion
  void limitCombinatorics(std::vector< MuonRecHitContainer > & MuonRecHitContainer_perLayer);
    
  std::vector<MuonRecHitContainer>
  findAllValidSets(const std::vector<MuonRecHitContainer> & MuonRecHitContainer_perLayer);

  std::pair <int, int> checkAngleDeviation(double dPhi_1, double dPhi_2) const;

  void validSetsPrePruning(std::vector<MuonRecHitContainer> & allValidSets);

  void pre_prune(MuonRecHitContainer & validSet) const;

  std::vector <SeedCandidate>
  fillSeedCandidates(std::vector <MuonRecHitContainer> & allValidSets);

  void estimateMomentum(const MuonRecHitContainer & validSet, 
                        CLHEP::Hep3Vector & momentum, int & charge) const;

  TrajectorySeed makeSeed(const TrajectoryStateOnSurface & tsos, 
                          const TransientTrackingRecHit::ConstRecHitContainer & hits) const;

private:
  MuonServiceProxy * theService;

  bool apply_prePruning;
  bool useSegmentsInTrajectory;


};

#endif



