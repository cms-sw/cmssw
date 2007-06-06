#ifndef RecoMuon_MuonSeedGenerator_MuonSeedGenerator_H
#define RecoMuon_MuonSeedGenerator_H

/** \class MuonSeedGenerator
 *  No description available.
 *
 *  $Date: 2006/11/13 08:55:24 $
 *  $Revision: 1.6 $
 *  \author R. Bellan - INFN Torino
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"

#include <vector>

namespace edm {class ParameterSet; class Event; class EventSetup;}

class MuonSeedFinder;

class MuonSeedGenerator: public edm::EDProducer {
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
  void complete(MuonSeedFinder& seed, MuonTransientTrackingRecHit::MuonRecHitContainer &recHits, bool* used=0) const;
  void checkAndFill(MuonSeedFinder& Theseed, const edm::EventSetup& eSetup);
  void fill(MuonSeedFinder& theSeed, const edm::EventSetup& eSetup);
  
  // FIXME: change in OwnVector?
  std::vector<TrajectorySeed> theSeeds;

  /// the name of the DT rec hits collection
  edm::InputTag theDTRecSegmentLabel;

  /// the name of the CSC rec hits collection
  edm::InputTag theCSCRecSegmentLabel;

  ///Enable the DT measurement
  bool enableDTMeasurement;

  ///Enable the CSC measurement
  bool enableCSCMeasurement;
};
#endif

