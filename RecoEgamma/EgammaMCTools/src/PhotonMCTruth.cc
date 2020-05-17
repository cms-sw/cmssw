
#include "RecoEgamma/EgammaMCTools/interface/PhotonMCTruth.h"
#include "RecoEgamma/EgammaMCTools/interface/ElectronMCTruth.h"

#include <iostream>

PhotonMCTruth::PhotonMCTruth(int isAConversion,
                             const CLHEP::HepLorentzVector& phoMom,
                             int vertIndex,
                             int trackId,
                             int motherId,
                             const CLHEP::HepLorentzVector& mothMom,
                             const CLHEP::HepLorentzVector& mothVtx,
                             const CLHEP::HepLorentzVector& convVertex,
                             const CLHEP::HepLorentzVector& pV,
                             std::vector<ElectronMCTruth>& electrons)
    : isAConversion_(isAConversion),
      thePhoton_(phoMom),
      theVertexIndex_(vertIndex),
      theTrackId_(trackId),
      theMotherId_(motherId),
      theMotherMom_(mothMom),
      theMotherVtx_(mothVtx),
      theConvVertex_(convVertex),
      thePrimaryVertex_(pV),
      theElectrons_(electrons) {}
