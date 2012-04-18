#ifndef RECOLOCALCALO_HFCLUSTERPRODUCER_HLTHFRECOECALCANDIDATEPRODUCER_H
#define RECOLOCALCALO_HFCLUSTERPRODUCER_HLTHFRECOECALCANDIDATEPRODUCER_H 1// -*- C++ -*-
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
#include "RecoEgamma/EgammaHFProducers/interface/HFValueStruct.h"

//#include "MagneticField/Engine/interface/MagneticField.h"

class HLTHFRecoEcalCandidateProducer : public edm::EDProducer {
public:
  explicit HLTHFRecoEcalCandidateProducer(edm::ParameterSet const& conf);
  virtual void produce(edm::Event& e, edm::EventSetup const& iSetup);
private:
  edm::InputTag hfclusters_;
   int HFDBversion_;
  std::vector<double> HFDBvector_;
  reco::HFValueStruct hfvars_;
  bool doPU_; 
  HFRecoEcalCandidateAlgo algo_;
  std::vector<double> defaultDB_; 
};

#endif
