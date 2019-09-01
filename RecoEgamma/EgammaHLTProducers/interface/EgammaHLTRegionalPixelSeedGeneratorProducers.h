#ifndef EgammaHLTRegionalPixelSeedGeneratorProducers_h
#define EgammaHLTRegionalPixelSeedGeneratorProducers_h

//
// Package:         RecoEgamma/EgammaHLTProducers
// Class:           EgammaHLTRegionalPixelSeedGeneratorProducers
//
// Description:     Calls RoadSeachSeedFinderAlgorithm
//                  to find TrajectorySeeds.

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

class SeedGeneratorFromRegionHits;

namespace edm {
  class ConfigurationDescriptions;
}

class EgammaHLTRegionalPixelSeedGeneratorProducers : public edm::stream::EDProducer<> {
public:
  explicit EgammaHLTRegionalPixelSeedGeneratorProducers(const edm::ParameterSet& conf);

  ~EgammaHLTRegionalPixelSeedGeneratorProducers() override;

  void produce(edm::Event& e, const edm::EventSetup& c) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  std::unique_ptr<SeedGeneratorFromRegionHits> combinatorialSeedGenerator;
  double ptmin_;
  double vertexz_;
  double originradius_;
  double halflength_;
  double originz_;
  double deltaEta_;
  double deltaPhi_;

  edm::EDGetTokenT<reco::RecoEcalCandidateCollection> candTag_;
  edm::EDGetTokenT<reco::ElectronCollection> candTagEle_;
  edm::EDGetTokenT<reco::BeamSpot> BSProducer_;

  bool useZvertex_;
};

#endif
