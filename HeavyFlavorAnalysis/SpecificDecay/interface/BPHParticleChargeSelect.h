#ifndef HeavyFlavorAnalysis_SpecificDecay_BPHParticleChargeSelect_h
#define HeavyFlavorAnalysis_SpecificDecay_BPHParticleChargeSelect_h
/** \class BPHParticleChargeSelect
 *
 *  Description: 
 *     Class for particle selection by charge
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoSelect.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

//---------------
// C++ Headers --
//---------------


//              ---------------------
//              -- Class Interface --
//              ---------------------

class BPHParticleChargeSelect: public BPHRecoSelect {

 public:

  /** Constructor
   */
  BPHParticleChargeSelect( int c ): charge( c ? ( c > 0 ? 1 : -1 ) : 0 ) {}

  /** Destructor
   */
  virtual ~BPHParticleChargeSelect() {}

  /** Operations
   */
  /// select particle
  virtual bool accept( const reco::Candidate& cand ) const {
    switch ( charge ) {
    default:
    case  0: return ( cand.charge() != 0 );
    case  1: return ( cand.charge() >  0 );
    case -1: return ( cand.charge() <  0 );
    }
    return true;
  };

  /// set seelction charge
  void setCharge( int c ) { charge =( c ? ( c > 0 ? 1 : -1 ) : 0 ); return; }

  /// get seelction charge
  double getCharge() const { return charge; }

 private:

  // private copy and assigment constructors
  BPHParticleChargeSelect           ( const BPHParticleChargeSelect& x );
  BPHParticleChargeSelect& operator=( const BPHParticleChargeSelect& x );

  int charge;

};


#endif

