#ifndef HeavyFlavorAnalysis_SpecificDecay_BPHBsToJPsiPhiBuilder_h
#define HeavyFlavorAnalysis_SpecificDecay_BPHBsToJPsiPhiBuilder_h
/** \class BPHBsToJPsiPhiBuilder
 *
 *  Description: 
 *     Class to build Bs to JPsi Phi candidates
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

class BPHBsToJPsiPhiBuilder {

 public:

  /** Constructor
   */
  BPHBsToJPsiPhiBuilder( const edm::EventSetup& es,
      const std::vector<BPHPlusMinusConstCandPtr>& jpsiCollection,
      const std::vector<BPHPlusMinusConstCandPtr>&  phiCollection );

  /** Destructor
   */
  virtual ~BPHBsToJPsiPhiBuilder();

  /** Operations
   */
  /// build Bs candidates
  std::vector<BPHRecoConstCandPtr> build();

  /// set cuts
  void setJPsiMassMin( double m  );
  void setJPsiMassMax( double m  );
  void setPhiMassMin ( double m  );
  void setPhiMassMax ( double m  );
  void setMassMin    ( double m  );
  void setMassMax    ( double m  );
  void setProbMin    ( double p  );
  void setMassFitMin ( double m  );
  void setMassFitMax ( double m  );
  void setConstr     ( bool flag );

  /// get current cuts
  double getJPsiMassMin() const;
  double getJPsiMassMax() const;
  double getPhiMassMin () const;
  double getPhiMassMax () const;
  double getMassMin    () const;
  double getMassMax    () const;
  double getProbMin    () const;
  double getMassFitMin () const;
  double getMassFitMax () const;
  bool   getConstr     () const;

 private:

  // private copy and assigment constructors
  BPHBsToJPsiPhiBuilder           ( const BPHBsToJPsiPhiBuilder& x );
  BPHBsToJPsiPhiBuilder& operator=( const BPHBsToJPsiPhiBuilder& x );

  std::string jPsiName;
  std::string  phiName;

  const edm::EventSetup* evSetup;
  const std::vector<BPHPlusMinusConstCandPtr>* jCollection;
  const std::vector<BPHPlusMinusConstCandPtr>* pCollection;

  BPHMassSelect   * jpsiSel;
  BPHMassSelect   * mphiSel;

  BPHMassSelect   * massSel;
  BPHChi2Select   * chi2Sel;
  BPHMassFitSelect* mFitSel;

  bool massConstr;
  float minPDiff;
  bool updated;

  std::vector<BPHRecoConstCandPtr> bsList;

};


#endif

