#ifndef PizeroMCTruth_h
#define PizeroMCTruth_h

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include <CLHEP/Matrix/Vector.h>
#include <CLHEP/Vector/LorentzVector.h>
#include <vector>



/** \class PizeroMCTruth
 *       
 *  This class stores all the MC truth information needed about the
 *  pi0 
 * 
 *  $Date: 2009/05/27 07:34:56 $
 *  $Revision  $
 *  \author N. Marinelli  University of Notre Dame
*/

class PhotonMCTruth;
class ElectronMCTruth;
class PizeroMCTruth {

  public:
    PizeroMCTruth();
    PizeroMCTruth( const CLHEP::HepLorentzVector&  pizMom, 
		   std::vector<PhotonMCTruth>& photons,
		   const CLHEP::HepLorentzVector& pV);  
		    


    CLHEP::HepLorentzVector fourMomentum() const {return thePizero_;} 
    CLHEP::HepLorentzVector primaryVertex() const {return thePrimaryVertex_;} 
    std::vector<PhotonMCTruth> photons() const { return thePhotons_;}
 


 private:
    CLHEP::HepLorentzVector thePizero_;
    std::vector<PhotonMCTruth> thePhotons_;
    CLHEP::HepLorentzVector thePrimaryVertex_;
    



};

#endif
