#ifndef ElectronMCTruth_h
#define ElectronMCTruth_h

#include "SimDataFormats/Track/interface/SimTrack.h"
#include <CLHEP/Matrix/Vector.h>
#include <CLHEP/Vector/LorentzVector.h>
#include <vector>

class ElectronMCTruth {

  public:
    ElectronMCTruth();
    void SetBrem(float r, float z, float phoFrac, float eGamma, float eElectron);

    float GetBremR();
    float GetBremZ();
    float GetBremFraction();
    float GetBremPhotonE();
    float GetBremElectronE();

  private:
    float r_;
    float z_;
    float phoFrac_;
    float eGamma_;
    float eElectron_;

};

#endif
