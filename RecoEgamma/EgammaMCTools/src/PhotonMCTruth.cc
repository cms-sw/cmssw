
#include "RecoEgamma/EgammaMCTools/interface/PhotonMCTruth.h"
#include "RecoEgamma/EgammaMCTools/interface/ElectronMCTruth.h"

#include <iostream>


PhotonMCTruth::PhotonMCTruth(int isAConversion,
			     HepLorentzVector phoMom,
                             int vertIndex, 
                             int trackId,
                             int motherId,
			     HepLorentzVector mothMom,
			     HepLorentzVector mothVtx,
			     HepLorentzVector convVertex,  
			     HepLorentzVector pV,  
			     std::vector<ElectronMCTruth>& electrons ) :
  isAConversion_(isAConversion),
  thePhoton_(phoMom), 
  theVertexIndex_(vertIndex),
  theTrackId_(trackId),
  theMotherId_(motherId),
  theMotherMom_(mothMom),
  theMotherVtx_(mothVtx),
  theConvVertex_(convVertex), 
  thePrimaryVertex_(pV), 
  theElectrons_(electrons)  {

  
}




