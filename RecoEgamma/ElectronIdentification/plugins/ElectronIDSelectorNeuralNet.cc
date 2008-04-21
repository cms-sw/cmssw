#include "RecoEgamma/ElectronIdentification/plugins/ElectronIDSelectorNeuralNet.h"

ElectronIDSelectorNeuralNet::ElectronIDSelectorNeuralNet (const edm::ParameterSet& conf) : conf_ (conf) 
{
  doNeuralNet_ = conf_.getParameter<bool> ("doNeuralNet");
  
  if (doNeuralNet_) 
    neuralNetAlgo_ = new ElectronNeuralNet();
}

ElectronIDSelectorNeuralNet::~ElectronIDSelectorNeuralNet () 
{
  if (doNeuralNet_) 
    delete neuralNetAlgo_ ;
}

void ElectronIDSelectorNeuralNet::newEvent (const edm::Event& e, const edm::EventSetup& c)
{
  if (doNeuralNet_) 
    neuralNetAlgo_->setup (conf_);
}

double ElectronIDSelectorNeuralNet::operator () (const reco::GsfElectron & electron, const edm::Event& event) 
{
  if (doNeuralNet_) 
  	return static_cast<double>(neuralNetAlgo_->result (& (electron), event)) ;
  return 0. ;
}
