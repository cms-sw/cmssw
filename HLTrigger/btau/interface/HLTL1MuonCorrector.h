#ifndef HLTL1MuonCorrector_h
#define HLTL1MuonCorrector_h

//
// Package:         HLTrigger/btau
// Class:           HLTMuonCorrector
// 
/**\class HLTL1MuonCorrector

Converts L1Muons into Candidates and corrects for phi-rotation in the 
 magnetic field

 Implementation:
     Input is an HLTFilterObjectWithRefs and output is a CandidateCollection
*/
// Description:     Produces a candidate from a HLTROWR with a phi correction
//					for the magnetic fiels

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

class HLTL1MuonCorrector : public edm::EDProducer
{
 public:

  explicit HLTL1MuonCorrector(const edm::ParameterSet& conf);

  virtual ~HLTL1MuonCorrector();

  virtual void produce(edm::Event& e, const edm::EventSetup& c);

 private:
  float ptmin;
  float originz;
  float thePhiCorrection;
  edm::InputTag directionSrc;
  std::string vertexSrc;
  double candMass;
};

#endif
