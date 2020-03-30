#ifndef HeavyFlavorAnalysis_SpecificDecay_BPHBdToKxMuMuBuilder_h
#define HeavyFlavorAnalysis_SpecificDecay_BPHBdToKxMuMuBuilder_h
/** \class BPHBdToKxMuMuBuilder
 *
 *  Description: 
 *     Class to build B0 to K*0 mu+ mu- candidates
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

class BPHBdToKxMuMuBuilder {
public:
  /** Constructor
   */
  BPHBdToKxMuMuBuilder(const edm::EventSetup& es,
                       const std::vector<BPHPlusMinusConstCandPtr>& oniaCollection,
                       const std::vector<BPHPlusMinusConstCandPtr>& kx0Collection);

  // deleted copy constructor and assignment operator
  BPHBdToKxMuMuBuilder(const BPHBdToKxMuMuBuilder& x) = delete;
  BPHBdToKxMuMuBuilder& operator=(const BPHBdToKxMuMuBuilder& x) = delete;

  /** Destructor
   */
  virtual ~BPHBdToKxMuMuBuilder();

  /** Operations
   */
  /// build Bs candidates
  std::vector<BPHRecoConstCandPtr> build();

  /// set cuts
  void setOniaMassMin(double m);
  void setOniaMassMax(double m);
  void setKxMassMin(double m);
  void setKxMassMax(double m);
  void setMassMin(double m);
  void setMassMax(double m);
  void setProbMin(double p);
  void setMassFitMin(double m);
  void setMassFitMax(double m);
  void setConstr(bool flag);

  /// get current cuts
  double getOniaMassMin() const;
  double getOniaMassMax() const;
  double getKxMassMin() const;
  double getKxMassMax() const;
  double getMassMin() const;
  double getMassMax() const;
  double getProbMin() const;
  double getMassFitMin() const;
  double getMassFitMax() const;
  bool getConstr() const;

private:
  std::string oniaName;
  std::string kx0Name;

  const edm::EventSetup* evSetup;
  const std::vector<BPHPlusMinusConstCandPtr>* jCollection;
  const std::vector<BPHPlusMinusConstCandPtr>* kCollection;

  BPHMassSelect* oniaSel;
  BPHMassSelect* mkx0Sel;

  BPHMassSelect* massSel;
  BPHChi2Select* chi2Sel;
  BPHMassFitSelect* mFitSel;

  bool massConstr;
  float minPDiff;
  bool updated;

  std::vector<BPHRecoConstCandPtr> bdList;
};

#endif
