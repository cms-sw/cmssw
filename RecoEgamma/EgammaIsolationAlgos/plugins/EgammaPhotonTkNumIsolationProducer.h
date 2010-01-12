#ifndef EgammaIsolationProducers_EgammaPhotonTkNumIsolationProducer_h
#define EgammaIsolationProducers_EgammaPhotonTkNumIsolationProducer_h

//*****************************************************************************
// File:      EgammaPhotonTkNumIsolationProducer.h
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

class EgammaPhotonTkNumIsolationProducer : public edm::EDProducer {
 public:
  explicit EgammaPhotonTkNumIsolationProducer(const edm::ParameterSet&);
  ~EgammaPhotonTkNumIsolationProducer();
  
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
