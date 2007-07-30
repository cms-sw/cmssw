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
 *  $Date: $
 *  $Revision  $
 *  \author N. Marinelli  University of Notre Dame
*/

class PhotonMCTruth;
class ElectronMCTruth;
class PizeroMCTruth {

  public:
    PizeroMCTruth();
    PizeroMCTruth( const HepLorentzVector&  pizMom, 
		   std::vector<PhotonMCTruth>& photons,
		   const HepLorentzVector& pV);  
		    


    HepLorentzVector fourMomentum() const {return thePizero_;} 
    HepLorentzVector primaryVertex() const {return thePrimaryVertex_;} 
    std::vector<PhotonMCTruth> photons() const { return thePhotons_;}
 


 private:
    HepLorentzVector thePizero_;
    std::vector<PhotonMCTruth> thePhotons_;
    HepLorentzVector thePrimaryVertex_;
    



};

#endif
