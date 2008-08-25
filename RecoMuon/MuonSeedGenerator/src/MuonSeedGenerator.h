#ifndef RecoMuon_MuonSeedGenerator_MuonSeedGenerator_H
#define RecoMuon_MuonSeedGenerator_H

/** \class MuonSeedGenerator
 *  No description available.
 *
 *  $Date: 2007/10/16 17:50:40 $
 *  $Revision: 1.10 $
 *  \author R. Bellan - INFN Torino
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"
#include "RecoMuon/MuonSeedGenerator/src/MuonSeedFinder.h"
#include <vector>


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
  void checkAndFill(MuonSeedFinder& seedFinder, const edm::EventSetup& eSetup);
  void fill(MuonSeedFinder& seedFinder, const edm::EventSetup& eSetup);
  //returns a zeroed array of bools of length listSize
  static bool * zero(unsigned listSize);

  MuonSeedFinder theSeedFinder;

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

