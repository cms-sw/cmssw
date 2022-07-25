#ifndef HeavyFlavorAnalysis_SpecificDecay_BPHDecayToResResBuilder_h
#define HeavyFlavorAnalysis_SpecificDecay_BPHDecayToResResBuilder_h
/** \class BPHDecayToResResBuilder
 *
 *  Description: 
 *     Base class to build a particle decaying to two particles, decaying
 *     themselves in cascade, for generic particle types
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecayToResResBuilderBase.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecayConstrainedBuilder.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecaySpecificBuilder.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "FWCore/Framework/interface/EventSetup.h"

class BPHEventSetupWrapper;

//---------------
// C++ Headers --
//---------------
#include <iostream>
#include <string>
#include <vector>

//              ---------------------
//              -- Class Interface --
//              ---------------------

template <class ProdType, class Res1Type, class Res2Type>
class BPHDecayToResResBuilder : public BPHDecayToResResBuilderBase,
                                public BPHDecayConstrainedBuilder<ProdType, Res1Type>,
                                public BPHDecaySpecificBuilder<ProdType> {
public:
  using typename BPHDecayGenericBuilder<ProdType>::prod_ptr;
  typedef typename Res1Type::const_pointer res1_ptr;
  typedef typename Res2Type::const_pointer res2_ptr;

  /** Constructor
   */
  BPHDecayToResResBuilder(const BPHEventSetupWrapper& es,
                          const std::string& res1Name,
                          double res1Mass,
                          double res1Width,
                          const std::vector<res1_ptr>& res1Collection,
                          const std::string& res2Name,
                          const std::vector<res2_ptr>& res2Collection)
      : BPHDecayGenericBuilderBase(es, nullptr),
        BPHDecayConstrainedBuilderBase(res1Name, res1Mass, res1Width),
        BPHDecayToResResBuilderBase(res2Name),
        BPHDecayConstrainedBuilder<ProdType, Res2Type>(res1Collection),
        sCollection(&res2Collection) {}

  // deleted copy constructor and assignment operator
  BPHDecayToResResBuilder(const BPHDecayToResResBuilder& x) = delete;
  BPHDecayToResResBuilder& operator=(const BPHDecayToResResBuilder& x) = delete;

  /** Destructor
   */
  ~BPHDecayToResResBuilder() override = default;

protected:
  BPHDecayToResResBuilder(const std::vector<res1_ptr>& res1Collection,
                          const std::string& res2Name,
                          const std::vector<res2_ptr>& res2Collection)
      : BPHDecayToResResBuilderBase(res2Name),
        BPHDecayConstrainedBuilder<ProdType, Res2Type>(res1Collection),
        sCollection(&res2Collection) {}

  const std::vector<res2_ptr>* sCollection;

  void addRes2Collection(BPHRecoBuilder& brb) override {
    const std::vector<res2_ptr>& sc = *this->sCollection;
    if (res2Sel->getMassMax() > 0.0) {
      sCollectSel.clear();
      sCollectSel.reserve(sc.size());
      for (const res2_ptr& s : sc) {
        if (res2Sel->accept(*s))
          sCollectSel.push_back(s);
      }
      brb.add(sName, sCollectSel);
    } else
      brb.add(sName, *this->sCollection);
  }

private:
  std::vector<res2_ptr> sCollectSel;
};

#endif
