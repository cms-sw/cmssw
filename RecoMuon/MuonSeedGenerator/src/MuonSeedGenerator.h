#ifndef RecoMuon_MuonSeedGenerator_MuonSeedGenerator_H
#define RecoMuon_MuonSeedGenerator_H

/** \class MuonSeedGenerator
 *  No description available.
 *
 *  $Date: 2006/03/24 11:43:48 $
 *  $Revision: 1.1 $
 *  \author R. Bellan - INFN Torino
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

#include <vector>

namespace edm {class ParameterSet; class Event; class EventSetup;}

class TrackingRecHit;
class MuonSeedFinder;

class MuonSeedGenerator: public edm::EDProducer {

  //FIXME
 public:
  typedef std::vector<TrackingRecHit>       RecHitContainer;
  typedef RecHitContainer::const_iterator   RecHitIterator;
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
  void complete(MuonSeedFinder& seed,RecHitContainer recHits, bool* used=0) const;
  void checkAndFill(MuonSeedFinder& Theseed);

  TrajectorySeedCollection theSeeds;

};
#endif

