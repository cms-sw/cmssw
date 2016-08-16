#ifndef BPHChi2Select_H
#define BPHChi2Select_H
/** \class BPHChi2Select
 *
 *  Description: 
 *     Class for candidate selection by chisquare (at vertex fit level)
 *
 *  $Date: 2016-05-03 14:47:26 $
 *  $Revision: 1.1 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHVertexSelect.h"
class BPHDecayVertex;

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------


//---------------
// C++ Headers --
//---------------


//              ---------------------
//              -- Class Interface --
//              ---------------------

class BPHChi2Select: public BPHVertexSelect {

 public:

  /** Constructor
   */
  BPHChi2Select( double prob );

  /** Destructor
   */
  virtual ~BPHChi2Select();

  /** Operations
   */
  /// select vertex
  virtual bool accept( const BPHDecayVertex& cand ) const;

  /// set prob min
  void setProbMin( double p );

  /// get current prob min
  double getProbMin() const;

 private:

  // private copy and assigment constructors
  BPHChi2Select           ( const BPHChi2Select& x );
  BPHChi2Select& operator=( const BPHChi2Select& x );

  double probMin;

};


#endif // BPHChi2Select_H

