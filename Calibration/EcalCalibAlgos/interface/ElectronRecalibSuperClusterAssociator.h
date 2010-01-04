#ifndef ElectronRecalibSuperClusterAssociator_h
#define ElectronRecalibSuperClusterAssociator_h
  
//
// Package:         RecoEgamma/EgammaElectronProducers
// Class:           ElectronRecalibSuperClusterAssociator
// 
// Description:   
  
  
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Common/interface/EDProduct.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>

class PixelMatchElectronAlgo;

class ElectronRecalibSuperClusterAssociator : public edm::EDProducer
{
 public:

  explicit ElectronRecalibSuperClusterAssociator(const edm::ParameterSet& conf);

  virtual ~ElectronRecalibSuperClusterAssociator();

  virtual void produce(edm::Event& e, const edm::EventSetup& c);

 private:

  std::string scProducer_;
  std::string scCollection_;
 
  std::string scIslandProducer_;
  std::string scIslandCollection_;
  
  std::string electronProducer_;
  std::string electronCollection_;

};
#endif
