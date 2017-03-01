#ifndef HeavyFlavorAnalysis_SpecificDecay_BPHParticleNeutralVeto_h
#define HeavyFlavorAnalysis_SpecificDecay_BPHParticleNeutralVeto_h
/** \class BPHParticleNeutralVeto
 *
 *  Description: 
 *     Class for neutral particle rejection
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

class BPHParticleNeutralVeto: public BPHRecoSelect {

 public:

  /** Constructor
   */
  BPHParticleNeutralVeto() {}

  /** Destructor
   */
  virtual ~BPHParticleNeutralVeto() {}

  /** Operations
   */
  /// select charged particles
  virtual bool accept( const reco::Candidate& cand ) const {
    return ( cand.charge() != 0 );
  }

 private:

  // private copy and assigment constructors
  BPHParticleNeutralVeto           ( const BPHParticleNeutralVeto& x );
  BPHParticleNeutralVeto& operator=( const BPHParticleNeutralVeto& x );

};


#endif

