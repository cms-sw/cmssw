#ifndef BPHMuonEtaSelect_H
#define BPHMuonEtaSelect_H
/** \class BPHMuonEtaSelect
 *
 *  Descrietaion: 
 *     Class for muon selection by eta
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

class BPHMuonEtaSelect: public BPHRecoSelect {

 public:

  /** Constructor
   */
  BPHMuonEtaSelect( double eta );

  /** Destructor
   */
  virtual ~BPHMuonEtaSelect();

  /** Operations
   */
  /// select muon
  virtual bool accept( const reco::Candidate& cand ) const;

  /// set eta max
  void setEtaMax( double eta );

  /// get current eta max
  double getEtaMax() const;

 private:

  // private copy and assigment constructors
  BPHMuonEtaSelect           ( const BPHMuonEtaSelect& x );
  BPHMuonEtaSelect& operator=( const BPHMuonEtaSelect& x );

  double etaMax;

};


#endif // BPHMuonEtaSelect_H

