#ifndef BPHMuonChargeSelect_H
#define BPHMuonChargeSelect_H
/** \class BPHMuonChargeSelect
 *
 *  Description: 
 *     Class for muon selection by charge
 *
 *
 *  $Date: 2016-05-03 14:57:34 $
 *  $Revision: 1.1 $
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


#endif // BPHMuonChargeSelect_H

