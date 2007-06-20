
#include "RecoEgamma/EgammaMCTools/interface/PhotonMCTruth.h"
#include "RecoEgamma/EgammaMCTools/interface/ElectronMCTruth.h"

#include <iostream>


PhotonMCTruth::PhotonMCTruth(int isAConversion,
			     HepLorentzVector v,
                             int vertIndex, 
                             int trackId,
			     HepLorentzVector convVertex,  
			     HepLorentzVector pV,  
			     std::vector<ElectronMCTruth>& electrons ) :
  isAConversion_(isAConversion),
  thePhoton_(v), 
  theVertexIndex_(vertIndex),
  theTrackId_(trackId),
  theConvVertex_(convVertex), 
  thePrimaryVertex_(pV), 
  theElectrons_(electrons)  {

  
}




