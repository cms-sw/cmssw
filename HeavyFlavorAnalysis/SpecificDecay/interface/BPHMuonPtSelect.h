#ifndef HeavyFlavorAnalysis_SpecificDecay_BPHMuonPtSelect_h
#define HeavyFlavorAnalysis_SpecificDecay_BPHMuonPtSelect_h
/** \class BPHMuonPtSelect
 *
 *  Description: 
 *     Class for muon selection by Pt
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHParticlePtSelect.h"

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

class BPHMuonPtSelect: public BPHParticlePtSelect {

 public:

  /** Constructor
   */
  BPHMuonPtSelect( double pt ): BPHParticlePtSelect( pt ) {}

  /** Destructor
   */
  virtual ~BPHMuonPtSelect() {}

  /** Operations
   */
  /// select muon
  virtual bool accept( const reco::Candidate& cand ) const {
    if ( dynamic_cast<const pat::Muon*>( &cand ) == 0 ) return false;
    return BPHParticlePtSelect::accept( cand );
  }

 private:

  // private copy and assigment constructors
  BPHMuonPtSelect           ( const BPHMuonPtSelect& x );
  BPHMuonPtSelect& operator=( const BPHMuonPtSelect& x );

};


#endif

