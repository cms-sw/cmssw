#ifndef HeavyFlavorAnalysis_SpecificDecay_BPHMuonEtaSelect_h
#define HeavyFlavorAnalysis_SpecificDecay_BPHMuonEtaSelect_h
/** \class BPHMuonEtaSelect
 *
 *  Descrietaion: 
 *     Class for muon selection by eta
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHParticleEtaSelect.h"

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

class BPHMuonEtaSelect: public BPHParticleEtaSelect {

 public:

  /** Constructor
   */
  BPHMuonEtaSelect( double eta ): BPHParticleEtaSelect( eta ) {}

  /** Destructor
   */
  virtual ~BPHMuonEtaSelect() {}

  /** Operations
   */
  /// select muon
  virtual bool accept( const reco::Candidate& cand ) const {
    if ( dynamic_cast<const pat::Muon*>( &cand ) == 0 ) return false;
    return BPHParticleEtaSelect::accept( cand );
  }

 private:

  // private copy and assigment constructors
  BPHMuonEtaSelect           ( const BPHMuonEtaSelect& x );
  BPHMuonEtaSelect& operator=( const BPHMuonEtaSelect& x );

};


#endif

