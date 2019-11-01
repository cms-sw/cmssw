#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronClassification.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"

//===================================================================
// Author: Federico Ferri - INFN Milano, Bicocca university
// 12/2005
// See GsfElectron::Classification
//===================================================================

using namespace reco;

GsfElectron::Classification egamma::classifyElectron(GsfElectron const& electron) {
  if (!electron.isEB() && !electron.isEE()) {
    edm::LogWarning("") << "ElectronClassification::init(): Undefined electron, eta = " << electron.eta() << "!!!!";
    return GsfElectron::UNKNOWN;
  }

  if (electron.isEBEEGap() || electron.isEBEtaGap() || electron.isEERingGap()) {
    return GsfElectron::GAP;
  }

  float fbrem = electron.trackFbrem();
  int nbrem = electron.numberOfBrems();

  if (electron.superClusterFbrem() - fbrem >= 0.15) {
    return GsfElectron::BADTRACK;
  }

  if (nbrem == 0 && fbrem < 0.5) {
    return GsfElectron::GOLDEN;
  }
  if (nbrem == 0 && fbrem >= 0.5) {
    return GsfElectron::BIGBREM;
  }
  return GsfElectron::SHOWERING;
}
