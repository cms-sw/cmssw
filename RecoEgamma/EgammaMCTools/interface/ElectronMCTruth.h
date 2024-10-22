#ifndef ElectronMCTruth_h
#define ElectronMCTruth_h

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include <CLHEP/Matrix/Vector.h>
#include <CLHEP/Vector/LorentzVector.h>
#include <vector>

/** \class ElectronMCTruth
 *       
 *  This class stores all the MC truth information needed about the
 *  electrons
 * 
 *  \author N. Marinelli  University of Notre Dame
*/

class ElectronMCTruth {
public:
  ElectronMCTruth();
  ElectronMCTruth(CLHEP::HepLorentzVector& v,
                  int vertIndex,
                  std::vector<CLHEP::Hep3Vector>& bremPos,
                  std::vector<CLHEP::HepLorentzVector>& pBrem,
                  std::vector<float>& xbrem,
                  CLHEP::HepLorentzVector& pV,
                  SimTrack& eTrack);

  CLHEP::HepLorentzVector fourMomentum() const { return theElectron_; }
  CLHEP::HepLorentzVector primaryVertex() const { return thePrimaryVertex_; }
  std::vector<CLHEP::Hep3Vector> bremVertices() const { return theBremPosition_; }
  std::vector<CLHEP::HepLorentzVector> bremMomentum() const { return theBremMomentum_; }
  std::vector<float> eloss() const { return theELoss_; }
  SimTrack simTracks() const { return eTrack_; }
  int vertexInd() const { return theVertexIndex_; }

private:
  CLHEP::HepLorentzVector theElectron_;
  int theVertexIndex_;
  std::vector<CLHEP::Hep3Vector> theBremPosition_;
  std::vector<CLHEP::HepLorentzVector> theBremMomentum_;
  std::vector<float> theELoss_;
  CLHEP::HepLorentzVector thePrimaryVertex_;
  SimTrack eTrack_;
};

#endif
