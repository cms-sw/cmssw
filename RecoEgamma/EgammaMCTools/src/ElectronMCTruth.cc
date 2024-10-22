#include "RecoEgamma/EgammaMCTools/interface/ElectronMCTruth.h"

#include <iostream>

ElectronMCTruth::ElectronMCTruth() {}

ElectronMCTruth::ElectronMCTruth(CLHEP::HepLorentzVector& v,
                                 int vertIndex,
                                 std::vector<CLHEP::Hep3Vector>& bremPos,
                                 std::vector<CLHEP::HepLorentzVector>& pBrem,
                                 std::vector<float>& xBrem,
                                 CLHEP::HepLorentzVector& pV,
                                 SimTrack& eTrack)
    :

      theElectron_(v),
      theVertexIndex_(vertIndex),
      theBremPosition_(bremPos),
      theBremMomentum_(pBrem),
      theELoss_(xBrem),
      thePrimaryVertex_(pV),
      eTrack_(eTrack) {}
