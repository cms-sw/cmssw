#include "RecoEgamma/EgammaMCTools/interface/ElectronMCTruth.h"

#include <iostream>

ElectronMCTruth::ElectronMCTruth() {

}


ElectronMCTruth::ElectronMCTruth( HepLorentzVector& v, 
				  std::vector<Hep3Vector>& bremPos, 
				  std::vector<HepLorentzVector>& pBrem, 
				  std::vector<float>& xBrem, 
				  HepLorentzVector& pV,  
				  SimTrack& eTrack ) :

      theElectron_(v), theBremPosition_(bremPos), theBremMomentum_(pBrem), theELoss_(xBrem),  thePrimaryVertex_(pV), eTrack_(eTrack) 
{

    }



 
