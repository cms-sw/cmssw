
#include "RecoEgamma/EgammaMCTools/interface/PhotonMCTruth.h"

#include <iostream>


PhotonMCTruth::PhotonMCTruth(int isAConversion,HepLorentzVector v, float rconv, float zconv,
					       HepLorentzVector convVertex,  
                                               HepLorentzVector pV,  std::vector<const SimTrack *> tracks  ) :
  isAConversion_(isAConversion),
  thePhoton_(v), theR_(rconv), theZ_(zconv), theConvVertex_(convVertex), 
  thePrimaryVertex_(pV), tracks_(tracks)  {

  //  std::cout << " PhotonMCTruth constructor " << std::endl; 

  
}
