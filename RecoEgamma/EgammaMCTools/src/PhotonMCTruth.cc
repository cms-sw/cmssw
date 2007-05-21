
#include "RecoEgamma/EgammaMCTools/interface/PhotonMCTruth.h"

#include <iostream>


PhotonMCTruth::PhotonMCTruth(int isAConversion,
			     HepLorentzVector v, 
			     HepLorentzVector convVertex,  
			     HepLorentzVector pV,  
			     std::vector<const SimTrack *> tracks  ) :
  isAConversion_(isAConversion),
  thePhoton_(v), 
  theConvVertex_(convVertex), 
  thePrimaryVertex_(pV), 
  tracks_(tracks)  {

  
}




