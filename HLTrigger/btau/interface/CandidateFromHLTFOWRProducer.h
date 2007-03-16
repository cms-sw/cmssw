#ifndef CandidateFromHLTFOWRProducer_h
#define CandidateFromHLTFOWRProducer_h

//
// Package:         RecoTracker/RegionalPixelSeedGenerator
// Class:           RegionalPixelSeedGenerator
// 
// Description:     Calls RoadSeachSeedFinderAlgorithm
//                  to find TrackingSeeds.


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/Common/interface/EDProduct.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoTracker/TkSeedGenerator/interface/CombinatorialRegionalSeedGeneratorFromPixel.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/JetReco/interface/Jet.h"

class CandidateFromHLTFOWRProducer : public edm::EDProducer
{
 public:

  explicit CandidateFromHLTFOWRProducer(const edm::ParameterSet& conf);

  virtual ~CandidateFromHLTFOWRProducer();

  virtual void produce(edm::Event& e, const edm::EventSetup& c);

 private:
  edm::ParameterSet conf_;
  float ptmin;
  float originz;
  float thePhiCorrection;
  edm::InputTag directionSrc;
  std::string vertexSrc;
  double candMass;
};

#endif
