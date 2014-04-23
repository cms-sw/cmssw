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

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "HFRecoEcalCandidateAlgo.h"
#include "HFValueStruct.h"

class HFRecoEcalCandidateProducer : public edm::stream::EDProducer<> {
 public:
  explicit HFRecoEcalCandidateProducer(edm::ParameterSet const& conf);
  virtual void produce(edm::Event& e, edm::EventSetup const& iSetup);
 private:
  std::vector<double> defaultDB_; 
  edm::InputTag hfclusters_,vertices_;
  int HFDBversion_;
  std::vector<double> HFDBvector_;
  bool doPU_; 
  double Cut2D_;
  double defaultSlope2D_;
  reco::HFValueStruct hfvars_;
  HFRecoEcalCandidateAlgo algo_;
};

#endif
