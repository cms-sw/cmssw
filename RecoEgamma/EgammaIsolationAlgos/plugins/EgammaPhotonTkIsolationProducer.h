#ifndef EgammaIsolationProducers_EgammaPhotonTkIsolationProducer_h
#define EgammaIsolationProducers_EgammaPhotonTkIsolationProducer_h

//*****************************************************************************
// File:      EgammaPhotonTkIsolationProducer.h
// ----------------------------------------------------------------------------
// OrigAuth:  Matthias Mozer
// Institute: IIHE-VUB
//=============================================================================
//*****************************************************************************

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

class EgammaPhotonTkIsolationProducer : public edm::EDProducer {
 public:
  explicit EgammaPhotonTkIsolationProducer(const edm::ParameterSet&);
  ~EgammaPhotonTkIsolationProducer();
  
  virtual void produce(edm::Event&, const edm::EventSetup&);

 private:
  edm::InputTag photonProducer_;
  edm::InputTag trackProducer_;
  edm::InputTag beamspotProducer_;

  double ptMin_;
  double intRadiusBarrel_;
  double intRadiusEndcap_;
  double stripBarrel_;
  double stripEndcap_;
  double extRadius_;
  double maxVtxDist_;
  double drb_;
  
  edm::ParameterSet conf_;

};


#endif
