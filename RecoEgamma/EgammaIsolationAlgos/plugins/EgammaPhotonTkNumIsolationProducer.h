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
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

class EgammaPhotonTkNumIsolationProducer : public edm::global::EDProducer<> {
public:
  explicit EgammaPhotonTkNumIsolationProducer(const edm::ParameterSet&);
  ~EgammaPhotonTkNumIsolationProducer() override;

  void produce(edm::StreamID sid, edm::Event&, const edm::EventSetup&) const override;

private:
  const edm::InputTag photonProducer_;
  const edm::InputTag trackProducer_;
  const edm::InputTag beamspotProducer_;

  const double ptMin_;
  const double intRadiusBarrel_;
  const double intRadiusEndcap_;
  const double stripBarrel_;
  const double stripEndcap_;
  const double extRadius_;
  const double maxVtxDist_;
  const double drb_;
};

#endif
