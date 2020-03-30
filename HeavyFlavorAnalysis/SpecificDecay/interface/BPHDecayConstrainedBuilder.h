#ifndef HeavyFlavorAnalysis_SpecificDecay_BPHDecayConstrainedBuilder_h
#define HeavyFlavorAnalysis_SpecificDecay_BPHDecayConstrainedBuilder_h
/** \class BPHDecayConstrainedBuilder
 *
 *  Description: 
 *     Class to build a particle decaying to a resonance, decaying itself
 *     to an opposite charged particles pair, applying a mass constraint
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecayGenericBuilder.h"

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

class BPHDecayConstrainedBuilder : public BPHDecayGenericBuilder {
public:
  /** Constructor
   */
  BPHDecayConstrainedBuilder(const edm::EventSetup& es,
                             const std::string& resName,
                             double resMass,
                             double resWidth,
                             const std::vector<BPHPlusMinusConstCandPtr>& resCollection);

  // deleted copy constructor and assignment operator
  BPHDecayConstrainedBuilder(const BPHDecayConstrainedBuilder& x) = delete;
  BPHDecayConstrainedBuilder& operator=(const BPHDecayConstrainedBuilder& x) = delete;

  /** Destructor
   */
  ~BPHDecayConstrainedBuilder() override;

  /** Operations
   */
  /// set cuts
  void setResMassMin(double m);
  void setResMassMax(double m);
  void setResMassRange(double mMin, double mMax);
  void setConstr(bool flag);

  /// get current cuts
  double getResMassMin() const { return resoSel->getMassMin(); }
  double getResMassMax() const { return resoSel->getMassMax(); }
  bool getConstr() const { return massConstr; }

protected:
  std::string rName;
  double rMass;
  double rWidth;

  const std::vector<BPHPlusMinusConstCandPtr>* rCollection;

  BPHMassSelect* resoSel;

  bool massConstr;
};

#endif
