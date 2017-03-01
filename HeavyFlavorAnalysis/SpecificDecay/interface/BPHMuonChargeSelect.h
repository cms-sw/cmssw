#ifndef HeavyFlavorAnalysis_SpecificDecay_BPHMuonChargeSelect_h
#define HeavyFlavorAnalysis_SpecificDecay_BPHMuonChargeSelect_h
/** \class BPHMuonChargeSelect
 *
 *  Description: 
 *     Class for muon selection by charge
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHParticleChargeSelect.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "DataFormats/PatCandidates/interface/Muon.h"

//---------------
// C++ Headers --
//---------------


//              ---------------------
//              -- Class Interface --
//              ---------------------

class BPHMuonChargeSelect: public BPHParticleChargeSelect {

 public:

  /** Constructor
   */
  BPHMuonChargeSelect( int c ): BPHParticleChargeSelect( c ) {}

  /** Destructor
   */
  virtual ~BPHMuonChargeSelect() {}

  /** Operations
   */
  /// select muon
  virtual bool accept( const reco::Candidate& cand ) const {
    if ( reinterpret_cast<const pat::Muon*>( &cand ) == 0 ) return false;
    return BPHParticleChargeSelect::accept( cand );
  };

 private:

  // private copy and assigment constructors
  BPHMuonChargeSelect           ( const BPHMuonChargeSelect& x );
  BPHMuonChargeSelect& operator=( const BPHMuonChargeSelect& x );

};


#endif

