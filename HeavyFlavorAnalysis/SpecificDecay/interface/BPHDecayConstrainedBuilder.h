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
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecayConstrainedBuilderBase.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoCandidate.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHPlusMinusCandidate.h"

#include "FWCore/Framework/interface/EventSetup.h"

class BPHEventSetupWrapper;

//---------------
// C++ Headers --
//---------------
#include <string>
#include <vector>

//              ---------------------
//              -- Class Interface --
//              ---------------------

template <class ProdType, class ResType>
class BPHDecayConstrainedBuilder : public virtual BPHDecayConstrainedBuilderBase,
                                   public virtual BPHDecayGenericBuilder<ProdType> {
public:
  using typename BPHDecayGenericBuilder<ProdType>::prod_ptr;
  typedef typename ResType::const_pointer res_ptr;

  /** Constructor
   */
  BPHDecayConstrainedBuilder(const BPHEventSetupWrapper& es,
                             const std::string& resName,
                             double resMass,
                             double resWidth,
                             const std::vector<res_ptr>& resCollection)
      : BPHDecayGenericBuilderBase(es),
        BPHDecayConstrainedBuilderBase(resName, resMass, resWidth),
        BPHDecayGenericBuilder<ProdType>(new BPHMassFitSelect(resName, resMass, resWidth, -2.0e+06, -1.0e+06)),
        rCollection(&resCollection) {}

  // deleted copy constructor and assignment operator
  BPHDecayConstrainedBuilder(const BPHDecayConstrainedBuilder& x) = delete;
  BPHDecayConstrainedBuilder& operator=(const BPHDecayConstrainedBuilder& x) = delete;

  /** Destructor
   */
  ~BPHDecayConstrainedBuilder() override = default;

protected:
  BPHDecayConstrainedBuilder(const std::vector<res_ptr>& resCollection) : rCollection(&resCollection) {}

  const std::vector<res_ptr>* rCollection;

  void addResCollection(BPHRecoBuilder& brb) override {
    const std::vector<res_ptr>& rc = *this->rCollection;
    if (resoSel->getMassMax() > 0.0) {
      rCollectSel.clear();
      rCollectSel.reserve(rc.size());
      for (const res_ptr& r : rc) {
        if (resoSel->accept(*r))
          rCollectSel.push_back(r);
      }
      brb.add(rName, rCollectSel);
    } else
      brb.add(rName, *this->rCollection);
  }

private:
  std::vector<res_ptr> rCollectSel;
};

#endif
