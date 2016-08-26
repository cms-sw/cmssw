#ifndef BPHMassSelect_H
#define BPHMassSelect_H
/** \class BPHMassSelect
 *
 *  Description: 
 *     Class for candidate selection by invariant mass (at momentum sum level)
 *
 *  $Date: 2016-05-03 14:47:26 $
 *  $Revision: 1.1 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHMomentumSelect.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHDecayMomentum.h"

//---------------
// C++ Headers --
//---------------


//              ---------------------
//              -- Class Interface --
//              ---------------------

class BPHMassSelect: public BPHMomentumSelect {

 public:

  /** Constructor
   */
  BPHMassSelect( double minMass, double maxMass ):
   mMin( minMass ),
   mMax( maxMass ) {
  }

  /** Destructor
   */
  virtual ~BPHMassSelect() {
  }

  /** Operations
   */
  /// select particle
  virtual bool accept( const BPHDecayMomentum& cand ) const {
    double mass = cand.composite().mass();
    return ( ( mass > mMin ) && ( mass < mMax ) );
  }

  /// set mass cuts
  void setMassMin( double m ) { mMin = m; return; }
  void setMassMax( double m ) { mMax = m; return; }

  /// get current mass cuts
  double getMassMin() const { return mMin; }
  double getMassMax() const { return mMax; }

 private:

  // private copy and assigment constructors
  BPHMassSelect           ( const BPHMassSelect& x );
  BPHMassSelect& operator=( const BPHMassSelect& x );

  double mMin;
  double mMax;

};


#endif // BPHMassSelect_H

