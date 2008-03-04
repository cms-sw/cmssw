#ifndef MuonSeedGenerator_CosmicMuonSeedGenerator_H
#define MuonSeedGenerator_CosmicMuonSeedGenerator_H

/** \class CosmicMuonSeedGenerator
 *  SeedGenerator for Cosmic Muon
 *
 *  $Date$
 *  $Revision$
 *  \author Chang Liu - Purdue University 
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"

#include <vector>

class MuonDetLayerGeometry;

namespace edm {class ParameterSet; class Event; class EventSetup;}


class CosmicMuonSeedGenerator: public edm::EDProducer {
 public:

  /// Constructor
  CosmicMuonSeedGenerator(const edm::ParameterSet&);
  
  /// Destructor
  virtual ~CosmicMuonSeedGenerator();
  
  // Operations

  /// reconstruct muon's seeds
  virtual void produce(edm::Event&, const edm::EventSetup&);

 private:

  /// generate TrajectorySeeds and put them into results
  void createSeeds(TrajectorySeedCollection& results,
                   const MuonTransientTrackingRecHit::MuonRecHitContainer& hits,
                   const edm::EventSetup& eSetup) const;

  /// determine if a MuonTransientTrackingRecHit is qualified to build seed
  bool checkQuality(const MuonTransientTrackingRecHit::MuonRecHitPointer&) const;

  /// select seed candidates from Segments in Event
  MuonTransientTrackingRecHit::MuonRecHitContainer  selectSegments(const MuonTransientTrackingRecHit::MuonRecHitContainer&) const;

  /// create TrajectorySeed from MuonTransientTrackingRecHit 
  std::vector<TrajectorySeed> createSeed(const MuonTransientTrackingRecHit::MuonRecHitPointer&,
                                         const edm::EventSetup&) const;

  /// compare direction of  two vectors
  float deltaEtaPhi(const GlobalVector&,const GlobalVector&) const;

  /// check if two rechits are correlated
  bool areCorrelated(const MuonTransientTrackingRecHit::MuonRecHitPointer&,
                     const MuonTransientTrackingRecHit::MuonRecHitPointer&) const;

  ///  compare quality of two rechits
  bool leftIsBetter(const MuonTransientTrackingRecHit::MuonRecHitPointer&,
                    const MuonTransientTrackingRecHit::MuonRecHitPointer&) const;


  struct DecreasingGlobalY{
    bool operator()(const MuonTransientTrackingRecHit::ConstMuonRecHitPointer &lhs,
		    const MuonTransientTrackingRecHit::ConstMuonRecHitPointer &rhs) const{ 
      return lhs->globalPosition().y() > rhs->globalPosition().y(); 
    }
  };

 private: 
  /// enable DT Segment Flag
  bool theEnableDTFlag;

  /// enable CSCSegment Flag
  bool theEnableCSCFlag;

  /// the name of the DT rec hits collection
  edm::InputTag theDTRecSegmentLabel;

  /// the name of the CSC rec hits collection
  edm::InputTag theCSCRecSegmentLabel;

  /// the maximum number of Seeds
  unsigned int theMaxSeeds;
  
  /// the maximum chi2 required for dt and csc rechits
  double theMaxDTChi2;
  double theMaxCSCChi2;
  edm::ESHandle<MuonDetLayerGeometry> theMuonLayers;
 
};
#endif

