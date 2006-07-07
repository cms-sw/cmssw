#ifndef MuonSeedGenerator_CosmicMuonSeedGenerator_H
#define MuonSeedGenerator_CosmicMuonSeedGenerator_H

/** \class CosmicMuonSeedGenerator
 *  SeedGenerator for Cosmic Muon
 *  some changes from MuonSeedGenerator 
 *  \author A. Vitelli, V.Palichik, ported by: R. Bellan 
 *  to cover more stations and have looser requirements
 *
 *  $Date: $
 *  $Revision: $
 *  \author Chang Liu - Purdue University 
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

#include <vector>

namespace edm {class ParameterSet; class Event; class EventSetup;}

class MuonSeedFinder;
class MuonTransientTrackingRecHit;

class CosmicMuonSeedGenerator: public edm::EDProducer {

 public:

  typedef std::vector<MuonTransientTrackingRecHit*>  RecHitContainer;
  typedef RecHitContainer::const_iterator            RecHitIterator;

 public:

  /// Constructor
  CosmicMuonSeedGenerator(const edm::ParameterSet&);
  
  /// Destructor
  virtual ~CosmicMuonSeedGenerator();
  
  // Operations

  /// reconstruct muon's seeds
  virtual void produce(edm::Event&, const edm::EventSetup&);

 protected:

 private:
  void complete(MuonSeedFinder& seed,RecHitContainer &recHits, bool* used=0) const;
  void checkAndFill(MuonSeedFinder& Theseed, const edm::EventSetup& eSetup);

  // FIXME: change in OwnVector?
  std::vector<TrajectorySeed> theSeeds;

  /// the name of the DT rec hits collection
  std::string theDTRecSegmentLabel;

  /// the name of the CSC rec hits collection
  std::string theCSCRecSegmentLabel;
};
#endif

