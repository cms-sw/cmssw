#include "RecoEgamma/EgammaMCTools/interface/PizeroMCTruth.h"
#include "RecoEgamma/EgammaMCTools/interface/PhotonMCTruth.h"
#include "RecoEgamma/EgammaMCTools/interface/ElectronMCTruth.h"


#include <iostream>

PizeroMCTruth::PizeroMCTruth() {

}


PizeroMCTruth::PizeroMCTruth( const CLHEP::HepLorentzVector& pizeroFourMomentum, 
			      std::vector<PhotonMCTruth>& photons,
			      const CLHEP::HepLorentzVector& pV):  
			      
  thePizero_(pizeroFourMomentum), thePhotons_(photons), thePrimaryVertex_(pV) {}




 
