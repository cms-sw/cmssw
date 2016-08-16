#ifndef BPHParticleChargeSelect_H
#define BPHParticleChargeSelect_H
/** \class BPHParticleChargeSelect
 *
 *  Description: 
 *     Class for particle selection by charge
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

class BPHParticleChargeSelect: public BPHRecoSelect {

 public:

  /** Constructor
   */
  BPHParticleChargeSelect( int c );

  /** Destructor
   */
  virtual ~BPHParticleChargeSelect();

  /** Operations
   */
  /// 
  virtual bool accept( const reco::Candidate& cand ) const;

 private:

  // private copy and assigment constructors
  BPHParticleChargeSelect           ( const BPHParticleChargeSelect& x );
  BPHParticleChargeSelect& operator=( const BPHParticleChargeSelect& x );

  int charge;

};


#endif // BPHParticleChargeSelect_H

