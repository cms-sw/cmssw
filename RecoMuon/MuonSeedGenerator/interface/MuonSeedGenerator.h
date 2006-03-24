#ifndef RecoMuon_MuonSeedGenerator_MuonSeedGenerator_H
#define RecoMuon_MuonSeedGenerator_H

/** \class MuonSeedGenerator
 *  No description available.
 *
 *  $Date: $
 *  $Revision: $
 *  \author R. Bellan - INFN Torino
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"


// FIXME!! It's dummy
#include "DataFormats/TrackReco/interface/RecHit.h" 
//was
//#include "CommonDet/BasicDet/interface/RecHit.h"
#include <vector>

namespace edm {class ParameterSet; class Event; class EventSetup;}

class MuonSeedGenerator: public edm::EDProducer {

  //FIXME
 public:
  typedef std::vector<RecHit>               RecHitContainer;
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

  TrajectorySeedCollection theSeeds;

};
#endif

