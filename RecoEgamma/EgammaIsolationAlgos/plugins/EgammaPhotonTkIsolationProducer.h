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
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

class EgammaPhotonTkIsolationProducer : public edm::global::EDProducer<> {
public:
  explicit EgammaPhotonTkIsolationProducer(const edm::ParameterSet&);
  ~EgammaPhotonTkIsolationProducer() override;

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

  const edm::ParameterSet conf_;
};

#endif
