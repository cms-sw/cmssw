#ifndef ElectronMCTruth_h
#define ElectronMCTruth_h

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include <CLHEP/Matrix/Vector.h>
#include <CLHEP/Vector/LorentzVector.h>
#include <vector>

class ElectronMCTruth {

  public:
    ElectronMCTruth();
    ElectronMCTruth( HepLorentzVector&  v, 
		     std::vector<Hep3Vector>& bremPos, 
		     std::vector<HepLorentzVector>& pBrem, 
		     std::vector<float>& xbrem, 
		     HepLorentzVector& pV,  
		     SimTrack& eTrack );


    HepLorentzVector fourMomentum() const {return theElectron_;} 
    HepLorentzVector primaryVertex() const {return thePrimaryVertex_;} 
    std::vector<Hep3Vector> bremVertices() const { return theBremPosition_;}
    std::vector<HepLorentzVector> bremMomentum() const { return theBremMomentum_;}
    std::vector<float>  eloss() const {return theELoss_;}
    SimTrack simTracks() const {return eTrack_;}     


 private:
    HepLorentzVector theElectron_;
    std::vector<Hep3Vector> theBremPosition_;
    std::vector<HepLorentzVector> theBremMomentum_;
    std::vector<float> theELoss_;
    HepLorentzVector thePrimaryVertex_;
    SimTrack eTrack_;



};

#endif
