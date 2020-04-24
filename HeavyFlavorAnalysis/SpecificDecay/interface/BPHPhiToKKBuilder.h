#ifndef HeavyFlavorAnalysis_SpecificDecay_BPHPhiToKKBuilder_h
#define HeavyFlavorAnalysis_SpecificDecay_BPHPhiToKKBuilder_h
/** \class BPHPhiToKKBuilder
 *
 *  Description: 
 *     Class to build Phi to K+ K- candidates
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

class BPHParticlePtSelect;
class BPHParticleEtaSelect;
class BPHChi2Select;
class BPHMassSelect;

//---------------
// C++ Headers --
//---------------
#include <string>
#include <vector>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class BPHPhiToKKBuilder {

 public:

  /** Constructor
   */
  BPHPhiToKKBuilder( const edm::EventSetup& es,
       const BPHRecoBuilder::BPHGenericCollection* kPosCollection,
       const BPHRecoBuilder::BPHGenericCollection* kNegCollection );

  /** Destructor
   */
  virtual ~BPHPhiToKKBuilder();

  /** Operations
   */
  /// build Phi candidates
  std::vector<BPHPlusMinusConstCandPtr> build();

  /// set cuts
  void setPtMin  ( double pt  );
  void setEtaMax ( double eta );
  void setMassMin( double m   );
  void setMassMax( double m   );
  void setProbMin( double p   );
  void setConstr ( double mass, double sigma );

  /// get current cuts
  double getPtMin  () const;
  double getEtaMax () const;
  double getMassMin() const;
  double getMassMax() const;
  double getProbMin() const;
  double getConstrMass () const;
  double getConstrSigma() const;

 private:

  // private copy and assigment constructors
  BPHPhiToKKBuilder           ( const BPHPhiToKKBuilder& x );
  BPHPhiToKKBuilder& operator=( const BPHPhiToKKBuilder& x );

  std::string kPosName;
  std::string kNegName;

  const edm::EventSetup* evSetup;
  const BPHRecoBuilder::BPHGenericCollection* posCollection;
  const BPHRecoBuilder::BPHGenericCollection* negCollection;

  BPHParticlePtSelect *  ptSel;
  BPHParticleEtaSelect* etaSel;
  BPHMassSelect* massSel;
  BPHChi2Select* chi2Sel;
  double cMass;
  double cSigma;
  bool updated;

  std::vector<BPHPlusMinusConstCandPtr> phiList;

};


#endif

