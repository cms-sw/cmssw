#ifndef HeavyFlavorAnalysis_SpecificDecay_BPHBuToJPsiKBuilder_h
#define HeavyFlavorAnalysis_SpecificDecay_BPHBuToJPsiKBuilder_h
/** \class BPHBuToJPsiKBuilder
 *
 *  Description: 
 *     Class to build B+- to JPsi K+- candidates
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------


//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoBuilder.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoCandidate.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHPlusMinusCandidate.h"

#include "FWCore/Framework/interface/Event.h"

class BPHParticleNeutralVeto;
class BPHParticlePtSelect;
class BPHParticleEtaSelect;
class BPHMassSelect;
class BPHChi2Select;
class BPHMassFitSelect;

//---------------
// C++ Headers --
//---------------
#include <string>
#include <vector>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class BPHBuToJPsiKBuilder {

 public:

  /** Constructor
   */
  BPHBuToJPsiKBuilder( const edm::EventSetup& es,
      const std::vector<BPHPlusMinusConstCandPtr>& jpsiCollection,
      const BPHRecoBuilder::BPHGenericCollection*  kaonCollection );

  /** Destructor
   */
  virtual ~BPHBuToJPsiKBuilder();

  /** Operations
   */
  /// build Bu candidates
  std::vector<BPHRecoConstCandPtr> build();

  /// set cuts
  void setKPtMin     ( double pt  );
  void setKEtaMax    ( double eta );
  void setJPsiMassMin( double m   );
  void setJPsiMassMax( double m   );
  void setMassMin    ( double m   );
  void setMassMax    ( double m   );
  void setProbMin    ( double p   );
  void setMassFitMin ( double m   );
  void setMassFitMax ( double m   );
  void setConstr     ( bool flag  );

  /// get current cuts
  double getKPtMin     () const;
  double getKEtaMax    () const;
  double getJPsiMassMin() const;
  double getJPsiMassMax() const;
  double getMassMin    () const;
  double getMassMax    () const;
  double getProbMin    () const;
  double getMassFitMin () const;
  double getMassFitMax () const;
  bool   getConstr     () const;

 private:

  // private copy and assigment constructors
  BPHBuToJPsiKBuilder           ( const BPHBuToJPsiKBuilder& x );
  BPHBuToJPsiKBuilder& operator=( const BPHBuToJPsiKBuilder& x );

  std::string jPsiName;
  std::string kaonName;

  const edm::EventSetup* evSetup;
  const std::vector<BPHPlusMinusConstCandPtr>* jCollection;
  const BPHRecoBuilder::BPHGenericCollection*  kCollection;

  BPHMassSelect         * jpsiSel;
  BPHParticleNeutralVeto* knVeto;
  BPHParticlePtSelect   *   ptSel;
  BPHParticleEtaSelect  *  etaSel;

  BPHMassSelect         * massSel;
  BPHChi2Select         * chi2Sel;
  BPHMassFitSelect      * mFitSel;

  bool massConstr;
  float minPDiff;
  bool updated;

  std::vector<BPHRecoConstCandPtr> buList;

};


#endif

