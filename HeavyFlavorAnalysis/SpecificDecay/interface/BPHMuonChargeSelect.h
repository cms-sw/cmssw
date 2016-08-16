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

class BPHMuonChargeSelect: public BPHRecoSelect {

 public:

  /** Constructor
   */
  BPHMuonChargeSelect( int c );

  /** Destructor
   */
  virtual ~BPHMuonChargeSelect();

  /** Operations
   */
  /// 
  virtual bool accept( const reco::Candidate& cand ) const;

 private:

  // private copy and assigment constructors
  BPHMuonChargeSelect           ( const BPHMuonChargeSelect& x );
  BPHMuonChargeSelect& operator=( const BPHMuonChargeSelect& x );

  int charge;

};


#endif // BPHMuonChargeSelect_H

