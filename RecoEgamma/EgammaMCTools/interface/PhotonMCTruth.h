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
 *  $Date: 2007/05/21 19:08:05 $
 *  $Revision: 1.5 $
 *  \author N. Marinelli  IASA-Athens
 *
 */




class PhotonMCTruth {
public:
  PhotonMCTruth() : isAConversion_(0),thePhoton_(0.,0.,0.), 
                       theConvVertex_(0.,0.,0.) {};

  PhotonMCTruth(HepLorentzVector v) : thePhoton_(v) {};


  PhotonMCTruth(int isAConversion,
		HepLorentzVector v,
		HepLorentzVector convVertex, 
		HepLorentzVector pV, 
		std::vector<const SimTrack *> tracks );
  

 HepLorentzVector primaryVertex() const {return thePrimaryVertex_;}
 int isAConversion() const { return isAConversion_;}
 HepLorentzVector fourMomentum() const {return thePhoton_;}
 HepLorentzVector vertex() const {return theConvVertex_;}
 std::vector<const SimTrack *> simTracks() const {return tracks_;} 
  
 private:

  int isAConversion_;
  HepLorentzVector thePhoton_;
  HepLorentzVector theConvVertex_;
  HepLorentzVector thePrimaryVertex_;
  std::vector<const SimTrack *> tracks_;

};

#endif

