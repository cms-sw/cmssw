#ifndef RECOLOCALCALO_HFCLUSTERPRODUCER_HLTHFRECOECALCANDIDATEPRODUCER_H
#define RECOLOCALCALO_HFCLUSTERPRODUCER_HLTHFRECOECALCANDIDATEPRODUCER_H 1  // -*- C++ -*-
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

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "HFRecoEcalCandidateAlgo.h"
#include "HFValueStruct.h"

class HLTHFRecoEcalCandidateProducer : public edm::global::EDProducer<> {
public:
  explicit HLTHFRecoEcalCandidateProducer(edm::ParameterSet const& conf);
  void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;

private:
  const edm::InputTag hfclusters_, vertices_;
  const int HFDBversion_;
  const std::vector<double> HFDBvector_;
  const double Cut2D_;
  const double defaultSlope2D_;
  const reco::HFValueStruct hfvars_;
  const HFRecoEcalCandidateAlgo algo_;
};

#endif
