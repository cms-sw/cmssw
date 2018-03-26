#include "RecoEgamma/EgammaMCTools/interface/ElectronMCTruth.h"

#include <iostream>

ElectronMCTruth::ElectronMCTruth() {

}


ElectronMCTruth::ElectronMCTruth( CLHEP::HepLorentzVector& v, 
				  int vertIndex,
                                  int hasBrem,
				  std::vector<CLHEP::Hep3Vector>& bremPos, 
				  std::vector<CLHEP::HepLorentzVector>& pBrem, 
				  std::vector<float>& xBrem, 
				  CLHEP::HepLorentzVector& pV,  
				  SimTrack& eTrack ) :
  
  theElectron_(v), 
  theVertexIndex_(vertIndex),
  hasBrem_(hasBrem), 
  theBremPosition_(bremPos), 
  theBremMomentum_(pBrem), 
  theELoss_(xBrem),  
  thePrimaryVertex_(pV), 
  eTrack_(eTrack) 
{
  
}



 
