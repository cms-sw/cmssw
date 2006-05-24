#ifndef RecoMuon_MuonSeedGenerator_MuonSeedGenerator_H
#define RecoMuon_MuonSeedGenerator_H

/** \class MuonSeedGenerator
 *  No description available.
 *
 *  $Date: 2006/05/15 17:25:28 $
 *  $Revision: 1.2 $
 *  \author R. Bellan - INFN Torino
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

#include <vector>

namespace edm {class ParameterSet; class Event; class EventSetup;}

class MuonSeedFinder;
class MuonTransientTrackingRecHit;

class MuonSeedGenerator: public edm::EDProducer {

 public:

  typedef std::vector<MuonTransientTrackingRecHit*>  RecHitContainer;
  typedef RecHitContainer::const_iterator            RecHitIterator;

 public:

  /// Constructor
  MuonSeedGenerator(const edm::ParameterSet&);
  
  /// Destructor
  virtual ~MuonSeedGenerator();
  
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

