#ifndef EgammaHLTRegionalPixelSeedGeneratorProducers_h
#define EgammaHLTRegionalPixelSeedGeneratorProducers_h

//
// Package:         RecoEgamma/EgammaHLTProducers
// Class:           EgammaHLTRegionalPixelSeedGeneratorProducers
// 
// Description:     Calls RoadSeachSeedFinderAlgorithm
//                  to find TrackingSeeds.


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/JetReco/interface/Jet.h"

class SeedGeneratorFromRegionHits;

class EgammaHLTRegionalPixelSeedGeneratorProducers : public edm::EDProducer
{
 public:

  explicit EgammaHLTRegionalPixelSeedGeneratorProducers(const edm::ParameterSet& conf);

  virtual ~EgammaHLTRegionalPixelSeedGeneratorProducers();

  virtual void produce(edm::Event& e, const edm::EventSetup& c);

  virtual void beginRun(edm::Run const&run, const edm::EventSetup& es) override final;
  virtual void endRun(edm::Run const&run, const edm::EventSetup& es) override final;


 private:
  edm::ParameterSet conf_;
  SeedGeneratorFromRegionHits *combinatorialSeedGenerator;
  double ptmin_;
  double vertexz_;
  double originradius_;
  double halflength_;
  double originz_;
  double deltaEta_;
  double deltaPhi_;
  edm::InputTag candTag_;
  edm::InputTag candTagEle_;
  bool useZvertex_;
  edm::InputTag BSProducer_;
};

#endif
