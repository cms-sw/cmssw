#ifndef BPHMuonPtSelect_H
#define BPHMuonPtSelect_H
/** \class BPHMuonPtSelect
 *
 *  Description: 
 *     Class for muon selection by Pt
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

class BPHMuonPtSelect: public BPHRecoSelect {

 public:

  /** Constructor
   */
  BPHMuonPtSelect( double pt );

  /** Destructor
   */
  virtual ~BPHMuonPtSelect();

  /** Operations
   */
  /// select muon
  virtual bool accept( const reco::Candidate& cand ) const;

  /// set pt min
  void setPtMin( double pt );

  /// get current pt min
  double getPtMin() const;

 private:

  // private copy and assigment constructors
  BPHMuonPtSelect           ( const BPHMuonPtSelect& x );
  BPHMuonPtSelect& operator=( const BPHMuonPtSelect& x );

  double ptMin;

};


#endif // BPHMuonPtSelect_H

