#ifndef PizeroMCTruth_h
#define PizeroMCTruth_h

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include <CLHEP/Matrix/Vector.h>
#include <CLHEP/Vector/LorentzVector.h>
#include "RecoEgamma/EgammaMCTools/interface/PhotonMCTruth.h"
#include <vector>



/** \class PizeroMCTruth
 *       
 *  This class stores all the MC truth information needed about the
 *  pi0 
 * 
 *  \author N. Marinelli  University of Notre Dame
*/

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
