#ifndef EgammaHLTPixelMatchElectronProducers_h
#define EgammaHLTPixelMatchElectronProducers_h

//
// Package:         RecoEgamma/EgammaHLTProducers
// Class:           EgammaHLTPixelMatchElectronProducers
//
// $Id: EgammaHLTPixelMatchElectronProducers.h,v 1.3 2009/10/14 14:32:23 covarell Exp $

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/EDPutToken.h"

#include "RecoEgamma/EgammaHLTAlgos/interface/EgammaHLTPixelMatchElectronAlgo.h"

#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"

#include <string>
#include <memory>

namespace edm {
  class ConfigurationDescriptions;
}

class EgammaHLTPixelMatchElectronAlgo;

class EgammaHLTPixelMatchElectronProducers : public edm::stream::EDProducer<> {
public:
  explicit EgammaHLTPixelMatchElectronProducers(const edm::ParameterSet& conf);

  void produce(edm::Event& e, const edm::EventSetup& c) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  EgammaHLTPixelMatchElectronAlgo algo_;
  const edm::EDPutTokenT<reco::ElectronCollection> token_;
};
#endif
