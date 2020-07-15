#ifndef HeavyFlavorAnalysis_SpecificDecay_BPHDecayToResResBuilder_h
#define HeavyFlavorAnalysis_SpecificDecay_BPHDecayToResResBuilder_h
/** \class BPHDecayToResResBuilder
 *
 *  Description: 
 *     Class to build a particle decaying to two resonances, decaying
 *     themselves to an opposite charged particles pair
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecayConstrainedBuilder.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoBuilder.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoCandidate.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHPlusMinusCandidate.h"

#include "FWCore/Framework/interface/Event.h"

//---------------
// C++ Headers --
//---------------
#include <string>
#include <vector>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class BPHDecayToResResBuilder : public BPHDecayConstrainedBuilder {
public:
  /** Constructor
   */
  BPHDecayToResResBuilder(const edm::EventSetup& es,
                          const std::string& res1Name,
                          double res1Mass,
                          double res1Width,
                          const std::vector<BPHPlusMinusConstCandPtr>& res1Collection,
                          const std::string& res2Name,
                          const std::vector<BPHPlusMinusConstCandPtr>& res2Collection);

  // deleted copy constructor and assignment operator
  BPHDecayToResResBuilder(const BPHDecayToResResBuilder& x) = delete;
  BPHDecayToResResBuilder& operator=(const BPHDecayToResResBuilder& x) = delete;

  /** Destructor
   */
  ~BPHDecayToResResBuilder() override;

  /** Operations
   */
  /// build candidates
  std::vector<BPHRecoConstCandPtr> build();

  /// set cuts
  void setRes1MassMin(double m) { setResMassMin(m); }
  void setRes1MassMax(double m) { setResMassMax(m); }
  void setRes1MassRange(double mMin, double mMax) { setResMassRange(mMin, mMax); }
  void setRes2MassMin(double m);
  void setRes2MassMax(double m);
  void setRes2MassRange(double mMin, double mMax);

  /// get current cuts
  double getRes1MassMin() const { return getResMassMin(); }
  double getRes1MassMax() const { return getResMassMax(); }
  double getRes2MassMin() const { return res2Sel->getMassMin(); }
  double getRes2MassMax() const { return res2Sel->getMassMax(); }

private:
  std::string sName;

  const std::vector<BPHPlusMinusConstCandPtr>* sCollection;

  BPHMassSelect* res2Sel;

  std::vector<BPHRecoConstCandPtr> recList;
};

#endif
