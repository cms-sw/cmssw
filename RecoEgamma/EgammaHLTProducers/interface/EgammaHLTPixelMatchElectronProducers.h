#ifndef EgammaHLTPixelMatchElectronProducers_h
#define EgammaHLTPixelMatchElectronProducers_h
  
//
// Package:         RecoEgamma/EgammaHLTProducers
// Class:           EgammaHLTPixelMatchElectronProducers
// 
// $Id: EgammaHLTPixelMatchElectronProducers.h,v 1.4 2011/05/20 17:17:28 wmtan Exp $
  
  
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"


#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>

class EgammaHLTPixelMatchElectronAlgo;

class EgammaHLTPixelMatchElectronProducers : public edm::EDProducer
{
 public:

  explicit EgammaHLTPixelMatchElectronProducers(const edm::ParameterSet& conf);

  virtual ~EgammaHLTPixelMatchElectronProducers();

  virtual void beginJob(void);
  virtual void produce(edm::Event& e, const edm::EventSetup& c);

 private:

  const edm::ParameterSet conf_;

  EgammaHLTPixelMatchElectronAlgo* algo_;
  std::string  seedProducer_;
};
#endif
