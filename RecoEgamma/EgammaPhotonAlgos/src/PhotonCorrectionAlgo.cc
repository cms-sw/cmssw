#include "RecoEgamma/EgammaPhotonAlgos/interface/PhotonCorrectionAlgo.h"
//


// Field
#include "MagneticField/Engine/interface/MagneticField.h"

// Geometry
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h" 
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
//



PhotonCorrectionAlgo::PhotonCorrectionAlgo() {

std::cout << " PhotonCorrectionAlgo CTOR  " << std::endl ; 

}

void PhotonCorrectionAlgo::makeCorrections(reco::Photon* pho) {

  std::cout << "  PhotonCorrectionAlgo::makeCorrections  " << std::endl;
  std::cout << "  Intial photon energy and position  " << pho->energy() << " " <<  pho->charge() << " " << pho->vertex() << std::endl;
  std::cout << "  SC energy " <<  pho->superCluster()->energy() << std::endl;
  std::cout << "  SC eta " <<  pho->superCluster()->position().eta() << std::endl;
 


}    
