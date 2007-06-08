#include "RecoEgamma/EgammaMCTools/interface/PizeroMCTruth.h"
#include "RecoEgamma/EgammaMCTools/interface/PhotonMCTruth.h"
#include "RecoEgamma/EgammaMCTools/interface/ElectronMCTruth.h"


#include <iostream>

PizeroMCTruth::PizeroMCTruth() {

}


PizeroMCTruth::PizeroMCTruth( const HepLorentzVector& pizeroFourMomentum, 
			      std::vector<PhotonMCTruth>& photons,
			      const HepLorentzVector& pV):  
			      
  thePizero_(pizeroFourMomentum), thePhotons_(photons), thePrimaryVertex_(pV) {}




 
