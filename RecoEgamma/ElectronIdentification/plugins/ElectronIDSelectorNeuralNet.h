#ifndef ElectronIDSelectorNeuralNet_h
#define ElectronIDSelectorNeuralNet_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "RecoEgamma/ElectronIdentification/interface/ElectronNeuralNet.h"

class ElectronIDSelectorNeuralNet
{
 public:

  explicit ElectronIDSelectorNeuralNet (const edm::ParameterSet& conf) ;
  virtual ~ElectronIDSelectorNeuralNet () ;

  void newEvent (const edm::Event&, const edm::EventSetup&) ;
  double operator() (const reco::GsfElectron&, const edm::Event&, const edm::EventSetup&) ;
   
 private:

  ElectronNeuralNet* neuralNetAlgo_;

  edm::ParameterSet conf_;
  
  bool doNeuralNet_;
  
};

#endif
