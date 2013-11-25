#include "RecoEgamma/ElectronIdentification/plugins/ElectronIDSelectorNeuralNet.h"

ElectronIDSelectorNeuralNet::ElectronIDSelectorNeuralNet (const edm::ParameterSet& conf, edm::ConsumesCollector & iC) : conf_ (conf)
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

void ElectronIDSelectorNeuralNet::newEvent (const edm::Event& e, const edm::EventSetup& es)
{
  if (doNeuralNet_)
    neuralNetAlgo_->setup (conf_);
}

double ElectronIDSelectorNeuralNet::operator () (const reco::GsfElectron & ele, const edm::Event& e, const edm::EventSetup& es)
{
  if (doNeuralNet_)
  	return static_cast<double>(neuralNetAlgo_->result (& (ele), e) );
  return 0. ;
}
