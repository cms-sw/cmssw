#ifndef RecoMuon_MuonSeedGenerator_MuonSeedGenerator_H
#define RecoMuon_MuonSeedGenerator_MuonSeedGenerator_H

/** \class MuonSeedGenerator
 *  No description available.
 *
 *  \author R. Bellan - INFN Torino
 */

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include <vector>
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

class MuonSeedVFinder;
class MuonSeedVPatternRecognition;
class MuonSeedVCleaner;
namespace edm {
  class ConfigurationDescriptions;
}

class MuonSeedGenerator: public edm::stream::EDProducer<> {
 public:

  /// Constructor
  MuonSeedGenerator(const edm::ParameterSet&);
  
  /// Destructor
  ~MuonSeedGenerator() override;
  
  // Operations

  /// reconstruct muon's seeds
  void produce(edm::Event&, const edm::EventSetup&) override;

  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
  
 protected:

  MuonSeedVPatternRecognition * thePatternRecognition;
  MuonSeedVFinder * theSeedFinder;
  MuonSeedVCleaner * theSeedCleaner;

  edm::InputTag theBeamSpotTag;
  edm::EDGetTokenT<reco::BeamSpot> beamspotToken;
};
#endif

