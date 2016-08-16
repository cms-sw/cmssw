#ifndef BPHParticleNeutralVeto_H
#define BPHParticleNeutralVeto_H
/** \class BPHParticleNeutralVeto
 *
 *  Description: 
 *     Class for neutral particle rejection
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
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoSelect.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------


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
  BPHParticleNeutralVeto();

  /** Destructor
   */
  virtual ~BPHParticleNeutralVeto();

  /** Operations
   */
  /// 
  virtual bool accept( const reco::Candidate& cand ) const;

 private:

  // private copy and assigment constructors
  BPHParticleNeutralVeto           ( const BPHParticleNeutralVeto& x );
  BPHParticleNeutralVeto& operator=( const BPHParticleNeutralVeto& x );

};


#endif // BPHParticleNeutralVeto_H

