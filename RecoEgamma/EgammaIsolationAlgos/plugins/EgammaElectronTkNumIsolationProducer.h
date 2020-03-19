#ifndef EgammaIsolationProducers_EgammaElectronTkNumIsolationProducer_h
#define EgammaIsolationProducers_EgammaElectronTkNumIsolationProducer_h

//*****************************************************************************
// File:      EgammaElectronTkNumIsolationProducer.h
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

class EgammaElectronTkNumIsolationProducer : public edm::global::EDProducer<> {
public:
  explicit EgammaElectronTkNumIsolationProducer(const edm::ParameterSet&);
  ~EgammaElectronTkNumIsolationProducer() override;

  void produce(edm::StreamID sid, edm::Event&, const edm::EventSetup&) const override;

private:
  const edm::InputTag electronProducer_;
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
