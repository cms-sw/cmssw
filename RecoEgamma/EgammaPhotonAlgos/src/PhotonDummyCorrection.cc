#include "RecoEgamma/EgammaPhotonAlgos/interface/PhotonDummyCorrection.h"
//
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
//



PhotonDummyCorrection::PhotonDummyCorrection() {

std::cout << " PhotonDummyCorrection CTOR  " << std::endl ; 

}

void PhotonDummyCorrection::makeCorrections(reco::Photon* pho) {

  using namespace edm;

  LogInfo(" PhotonDummyCorrection::makeCorrections  ") << "\n";
  LogInfo("  Photon energy and position  " ) << pho->energy() << " " << pho->vertex() << "\n";
 


}    
