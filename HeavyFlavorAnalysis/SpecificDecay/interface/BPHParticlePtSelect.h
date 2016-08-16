#ifndef BPHParticlePtSelect_H
#define BPHParticlePtSelect_H
/** \class BPHParticlePtSelect
 *
 *  Description: 
 *     Class for particle selection by Pt
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

class BPHParticlePtSelect: public BPHRecoSelect {

 public:

  /** Constructor
   */
  BPHParticlePtSelect( double pt );

  /** Destructor
   */
  virtual ~BPHParticlePtSelect();

  /** Operations
   */
  /// select particle
  virtual bool accept( const reco::Candidate& cand ) const;

  /// set pt min
  void setPtMin( double pt );

  /// get current pt min
  double getPtMin() const;

 private:

  // private copy and assigment constructors
  BPHParticlePtSelect           ( const BPHParticlePtSelect& x );
  BPHParticlePtSelect& operator=( const BPHParticlePtSelect& x );

  double ptMin;

};


#endif // BPHParticlePtSelect_H

