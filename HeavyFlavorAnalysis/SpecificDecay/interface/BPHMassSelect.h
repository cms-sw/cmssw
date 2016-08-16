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
class BPHDecayMomentum;

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------


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
  BPHMassSelect( double minMass, double maxMass );

  /** Destructor
   */
  virtual ~BPHMassSelect();

  /** Operations
   */
  /// accept particle
  virtual bool accept( const BPHDecayMomentum& cand ) const;

  /// set mass cuts
  void setMassMin( double m );
  void setMassMax( double m );

  /// get current mass cuts
  double getMassMin() const;
  double getMassMax() const;

 private:

  // private copy and assigment constructors
  BPHMassSelect           ( const BPHMassSelect& x );
  BPHMassSelect& operator=( const BPHMassSelect& x );

  double mMin;
  double mMax;

};


#endif // BPHMassSelect_H

