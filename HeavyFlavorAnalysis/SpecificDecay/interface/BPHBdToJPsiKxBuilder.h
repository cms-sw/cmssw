#ifndef HeavyFlavorAnalysis_SpecificDecay_BPHBdToJPsiKxBuilder_h
#define HeavyFlavorAnalysis_SpecificDecay_BPHBdToJPsiKxBuilder_h
/** \class BPHBdToJPsiKxBuilder
 *
 *  Description: 
 *     Class to build B0 to JPsi K*0 candidates
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

class BPHBdToJPsiKxBuilder {

 public:

  /** Constructor
   */
  BPHBdToJPsiKxBuilder( const edm::EventSetup& es,
      const std::vector<BPHPlusMinusConstCandPtr>& jpsiCollection,
      const std::vector<BPHPlusMinusConstCandPtr>&  kx0Collection );

  /** Destructor
   */
  virtual ~BPHBdToJPsiKxBuilder();

  /** Operations
   */
  /// build Bs candidates
  std::vector<BPHRecoConstCandPtr> build();

  /// set cuts
  void setJPsiMassMin( double m  );
  void setJPsiMassMax( double m  );
  void setKxMassMin  ( double m  );
  void setKxMassMax  ( double m  );
  void setMassMin    ( double m  );
  void setMassMax    ( double m  );
  void setProbMin    ( double p  );
  void setMassFitMin ( double m  );
  void setMassFitMax ( double m  );
  void setConstr     ( bool flag );

  /// get current cuts
  double getJPsiMassMin() const;
  double getJPsiMassMax() const;
  double getKxMassMin  () const;
  double getKxMassMax  () const;
  double getMassMin    () const;
  double getMassMax    () const;
  double getProbMin    () const;
  double getMassFitMin () const;
  double getMassFitMax () const;
  bool   getConstr     () const;

 private:

  // private copy and assigment constructors
  BPHBdToJPsiKxBuilder           ( const BPHBdToJPsiKxBuilder& x );
  BPHBdToJPsiKxBuilder& operator=( const BPHBdToJPsiKxBuilder& x );

  std::string jPsiName;
  std::string  kx0Name;

  const edm::EventSetup* evSetup;
  const std::vector<BPHPlusMinusConstCandPtr>* jCollection;
  const std::vector<BPHPlusMinusConstCandPtr>* kCollection;

  BPHMassSelect   * jpsiSel;
  BPHMassSelect   * mkx0Sel;

  BPHMassSelect   * massSel;
  BPHChi2Select   * chi2Sel;
  BPHMassFitSelect* mFitSel;

  bool massConstr;
  float minPDiff;
  bool updated;

  std::vector<BPHRecoConstCandPtr> bdList;

};


#endif

