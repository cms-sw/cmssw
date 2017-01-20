#ifndef HeavyFlavorAnalysis_SpecificDecay_BPHParticleEtaSelect_h
#define HeavyFlavorAnalysis_SpecificDecay_BPHParticleEtaSelect_h
/** \class BPHParticleEtaSelect
 *
 *  Descrietaion: 
 *     Class for particle selection by eta
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

class BPHParticleEtaSelect: public BPHRecoSelect {

 public:

  /** Constructor
   */
  BPHParticleEtaSelect( double eta ): etaMax( eta ) {}

  /** Destructor
   */
  virtual ~BPHParticleEtaSelect() {}

  /** Operations
   */
  /// select particle
  virtual bool accept( const reco::Candidate& cand ) const {
    return ( fabs( cand.p4().eta() ) < etaMax );
  }

  /// set eta max
  void setEtaMax( double eta ) { etaMax = eta; return; }

  /// get current eta max
  double getEtaMax() const { return etaMax; }

 private:

  // private copy and assigment constructors
  BPHParticleEtaSelect           ( const BPHParticleEtaSelect& x );
  BPHParticleEtaSelect& operator=( const BPHParticleEtaSelect& x );

  double etaMax;

};


#endif

