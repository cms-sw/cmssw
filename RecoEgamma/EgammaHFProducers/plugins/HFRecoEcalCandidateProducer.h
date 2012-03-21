#ifndef RECOLOCALCALO_HFCLUSTERPRODUCER_HFRECOECALCANDIDATEPRODUCER_H
#define RECOLOCALCALO_HFCLUSTERPRODUCER_HFRECOECALCANDIDATEPRODUCER_H 1// -*- C++ -*-
//
// Package:    EgammaHFProducers
// Class:      HFRecoEcalCandidateProducers
// 
/**\class HFRecoEcalCandidateProducers.h HFRecoEcalCandidateProducers.cc  
*/
//
// Original Author:  Kevin Klapoetke University of Minnesota
//         Created:  Wed 26 Sept 2007
// $Id:
//
//

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "RecoEgamma/EgammaHFProducers/interface/HFRecoEcalCandidateAlgo.h"
//#include "MagneticField/Engine/interface/MagneticField.h"

class HFRecoEcalCandidateProducer : public edm::EDProducer {
public:
  explicit HFRecoEcalCandidateProducer(edm::ParameterSet const& conf);
  virtual void produce(edm::Event& e, edm::EventSetup const& iSetup);
private:
  edm::InputTag hfclusters_;
  bool CorrectForPileup_;
  HFRecoEcalCandidateAlgo algo_;
  std::vector<double> defaultSlope_;
  std::vector<double> defaultIntercept_; 
};

#endif
