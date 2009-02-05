#ifndef RecoMuon_MuonSeedGenerator_SETMuonSeedProducer_H
#define RecoMuon_MuonSeedGenerator_SETMuonSeedProducer_H

/** \class SETMuonSeedProducer 
     I. Bloch, E. James, S. Stoynev
  */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"


#include "RecoMuon/TrackingTools/interface/RecoMuonEnumerators.h"


#include <RecoMuon/TrackingTools/interface/MuonServiceProxy.h>

#include "RecoMuon/MuonSeedGenerator/src/SETFilter.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h" 
#include "RecoMuon/MuonSeedGenerator/src/MuonSeedPtExtractor.h"

#include "FWCore/Framework/interface/Event.h"
#include "RecoMuon/TrackingTools/interface/MuonCandidate.h"

typedef MuonTransientTrackingRecHit::MuonRecHitContainer MuonRecHitContainer;
typedef std::vector<Trajectory*> TrajectoryContainer;

class TrajectorySeed;
class STAFilter;
class MuonServiceProxy;

//namespace edm {class ParameterSet;}
namespace edm {class ParameterSet; class Event; class EventSetup;}

class SETMuonSeedProducer : public edm::EDProducer {
  
 public:
  /// Constructor with Parameter set 
  SETMuonSeedProducer (const edm::ParameterSet&);
  
  /// Destructor
  virtual ~SETMuonSeedProducer();
  
  // Operations
  virtual void produce(edm::Event&, const edm::EventSetup&);
  
 protected:

 private:
  
  
  // Returns a vector of measurements sets (for later trajectory seed building)
  std::vector < std::pair < TrajectoryStateOnSurface, 
    TransientTrackingRecHit::ConstRecHitContainer > > trajectories(const edm::Event&);

  /// pre-filter
  SETFilter* filter() const {return theFilter;}
  
  /// pT extractor (given two hits)
  MuonSeedPtExtractor::MuonSeedPtExtractor* pt_extractor() const {return thePtExtractor;}

  //---- SET 
  /// Build local clusters of segments that are clearly separated from each other in the eta-phi plane 
  std::vector< MuonRecHitContainer > clusterHits( 
						 MuonRecHitContainer muonRecHits,
						 MuonRecHitContainer muonRecHits_DT2D_hasPhi,
						 MuonRecHitContainer muonRecHits_DT2D_hasZed,
						 MuonRecHitContainer muonRecHits_RPC);
  
  std::vector <MuonRecHitContainer> findAllValidSets(std::vector< MuonRecHitContainer > MuonRecHitContainer_perLayer);

  void validSetsPrePruning(std::vector <MuonRecHitContainer>  & allValidSets); 

  std::pair <int, int> checkAngleDeviation(double dPhi_1, double dPhi_2);

  std::vector <seedSet>  fillSeedSets(std::vector <MuonRecHitContainer> & allValidSets);
  //----

  //private:
  
  SETFilter* theFilter;
  void setEvent(const edm::Event&);
 
  //---- SET
  MuonSeedPtExtractor::MuonSeedPtExtractor* thePtExtractor;
  bool apply_prePruning;
  bool useSegmentsInTrajectory;
  bool useRPCs;

  edm::InputTag DTRecSegmentLabel;
  edm::InputTag CSCRecSegmentLabel;
  edm::InputTag RPCRecSegmentLabel;

  MuonServiceProxy *theService;
};
#endif
