#include "RecoEgamma/ElectronIdentification/interface/ElectronNeuralNet.h"

double ElectronNeuralNet::result(const reco::GsfElectron* electron,
				 const edm::Event& e) {
  return 1.;
}
