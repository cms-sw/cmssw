#ifndef PhotonMCTruth_h
#define PhotonMCTruth_h

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include <CLHEP/Matrix/Vector.h>
#include <CLHEP/Vector/LorentzVector.h>
#include <vector>

/** \class PhotonMCTruth
 *       
 *  This class stores all the MC truth information needed about the
 *  conversion
 * 
 *  $Date: 2009/05/27 07:34:56 $
 *  $Revision: 1.9 $
 *  \author N. Marinelli  University of Notre Dame
 *
 */



class ElectronMCTruth;
class PhotonMCTruth {
public:
  PhotonMCTruth() : isAConversion_(0),thePhoton_(0.,0.,0.), 
                       theConvVertex_(0.,0.,0.) {};

  PhotonMCTruth(CLHEP::HepLorentzVector v) : thePhoton_(v) {};


  PhotonMCTruth(int isAConversion,
		CLHEP::HepLorentzVector v,
                int vertIndex,
                int trackId,
                int motherId,
		CLHEP::HepLorentzVector mothMom, 
		CLHEP::HepLorentzVector mothVtx, 
		CLHEP::HepLorentzVector convVertex, 
		CLHEP::HepLorentzVector pV, 
		std::vector<ElectronMCTruth>& electrons );
  

 CLHEP::HepLorentzVector primaryVertex() const {return thePrimaryVertex_;}
 int isAConversion() const { return isAConversion_;}
 CLHEP::HepLorentzVector fourMomentum() const {return thePhoton_;}
 int vertexInd() const {return theVertexIndex_;}
 CLHEP::HepLorentzVector vertex() const {return theConvVertex_;}
 std::vector<ElectronMCTruth> electrons() const {return theElectrons_;} 
 int trackId() const {return theTrackId_;}  
 int motherType() const {return theMotherId_;}
 CLHEP::HepLorentzVector motherMomentum() const {return theMotherMom_;}
 CLHEP::HepLorentzVector motherVtx() const {return theMotherVtx_;}

 private:

  int isAConversion_;
  CLHEP::HepLorentzVector thePhoton_;
  int theVertexIndex_;
  int theTrackId_;
  int theMotherId_;
  CLHEP::HepLorentzVector theMotherMom_;
  CLHEP::HepLorentzVector theMotherVtx_;
  CLHEP::HepLorentzVector theConvVertex_;
  CLHEP::HepLorentzVector thePrimaryVertex_;
  std::vector<ElectronMCTruth> theElectrons_;

};

#endif

