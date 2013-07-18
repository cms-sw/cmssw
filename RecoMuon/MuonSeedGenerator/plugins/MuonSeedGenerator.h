#ifndef RecoMuon_MuonSeedGenerator_MuonSeedGenerator_H
#define RecoMuon_MuonSeedGenerator_H

/** \class MuonSeedGenerator
 *  No description available.
 *
 *  $Date: 2008/10/17 22:14:56 $
 *  $Revision: 1.1 $
 *  \author R. Bellan - INFN Torino
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include <vector>

class MuonSeedVFinder;
class MuonSeedVPatternRecognition;
class MuonSeedVCleaner;

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

  MuonSeedVPatternRecognition * thePatternRecognition;
  MuonSeedVFinder * theSeedFinder;
  MuonSeedVCleaner * theSeedCleaner;

  edm::InputTag theBeamSpotTag;

};
#endif

