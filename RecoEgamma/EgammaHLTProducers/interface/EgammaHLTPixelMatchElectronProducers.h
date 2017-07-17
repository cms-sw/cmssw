#ifndef EgammaHLTPixelMatchElectronProducers_h
#define EgammaHLTPixelMatchElectronProducers_h
  
//
// Package:         RecoEgamma/EgammaHLTProducers
// Class:           EgammaHLTPixelMatchElectronProducers
// 
// $Id: EgammaHLTPixelMatchElectronProducers.h,v 1.3 2009/10/14 14:32:23 covarell Exp $
  
  
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"


#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>

namespace edm {
  class ConfigurationDescriptions;
}

class EgammaHLTPixelMatchElectronAlgo;

class EgammaHLTPixelMatchElectronProducers : public edm::global::EDProducer<> {

 public:

  explicit EgammaHLTPixelMatchElectronProducers(const edm::ParameterSet& conf);
  ~EgammaHLTPixelMatchElectronProducers();

  void produce(edm::StreamID sid, edm::Event& e, const edm::EventSetup& c) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

 private:

  const edm::ParameterSet conf_;

  EgammaHLTPixelMatchElectronAlgo* algo_;
  std::string  seedProducer_;
};
#endif
