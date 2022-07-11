#ifndef HeavyFlavorAnalysis_SpecificDecay_BPHDecayToFlyingCascadeBuilder_h
#define HeavyFlavorAnalysis_SpecificDecay_BPHDecayToFlyingCascadeBuilder_h
/** \class BPHDecayToFlyingCascadeBuilder
 *
 *  Description: 
 *     Class to build a particle having a flying particle in the
 *     final state, for generic particle types
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecayToFlyingCascadeBuilderBase.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecayGenericBuilder.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHKinFitChi2Select.h"

#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoBuilder.h"

#include "FWCore/Framework/interface/EventSetup.h"

class BPHEventSetupWrapper;

//---------------
// C++ Headers --
//---------------
#include <string>
#include <vector>
#include <iostream>
//              ---------------------
//              -- Class Interface --
//              ---------------------

template <class ProdType, class FlyingType>
class BPHDecayToFlyingCascadeBuilder : public virtual BPHDecayToFlyingCascadeBuilderBase,
                                       public virtual BPHDecayGenericBuilder<ProdType> {
public:
  using typename BPHDecayGenericBuilder<ProdType>::prod_ptr;
  typedef typename FlyingType::const_pointer flying_ptr;

  /** Constructor
   */
  BPHDecayToFlyingCascadeBuilder(const BPHEventSetupWrapper& es,
                                 const std::string& flyName,
                                 double flyMass,
                                 double flyMSigma,
                                 const std::vector<flying_ptr>& flyCollection)
      : BPHDecayGenericBuilderBase(es, nullptr),
        BPHDecayToFlyingCascadeBuilderBase(flyName, flyMass, flyMSigma),
        fCollection(&flyCollection) {}

  // deleted copy constructor and assignment operator
  BPHDecayToFlyingCascadeBuilder(const BPHDecayToFlyingCascadeBuilder& x) = delete;
  BPHDecayToFlyingCascadeBuilder& operator=(const BPHDecayToFlyingCascadeBuilder& x) = delete;

  /** Destructor
   */
  ~BPHDecayToFlyingCascadeBuilder() override = default;

protected:
  BPHDecayToFlyingCascadeBuilder(const std::vector<flying_ptr>& flyCollection) : fCollection(&flyCollection) {}

  const std::vector<flying_ptr>* fCollection;

  void addFlyCollection(BPHRecoBuilder& brb) override {
    const std::vector<flying_ptr>& fc = *this->fCollection;
    if (flySel->getMassMax() > 0.0) {
      fCollectSel.clear();
      fCollectSel.reserve(fc.size());
      for (const flying_ptr& f : fc) {
        if (flySel->accept(*f))
          fCollectSel.push_back(f);
      }
      brb.add(fName, fCollectSel);
    } else
      brb.add(fName, *this->fCollection);
  }

  /// fit and select candidates
  void fitAndFilter(std::vector<prod_ptr>& prodList) {
    std::vector<prod_ptr> tempList;
    int iRec;
    int nRec = prodList.size();
    tempList.reserve(nRec);
    for (iRec = 0; iRec < nRec; ++iRec) {
      prod_ptr& ctmp = prodList[iRec];
      ProdType* cptr = ctmp->clone();
      prod_ptr cand(cptr);
      // fit for flying reconstruction
      // indipendent from other particles
      cptr->setIndependentFit(fName, true, fMass, fMSigma);
      cptr->resetKinematicFit();
      if ((mFitSel->getMassMax() >= 0) && (!mFitSel->accept(*cptr)))
        continue;
      const RefCountedKinematicVertex tdv = cptr->topDecayVertex();
      if ((kfChi2Sel->getProbMin() >= 0) && !kfChi2Sel->accept(*cptr))
        continue;
      const std::vector<std::string>& cList = ctmp->compNames();
      int iComp;
      int nComp = cList.size();
      for (iComp = 0; iComp < nComp; ++iComp) {
        const std::string& cName = cList[iComp];
        dMap[cand->getComp(cName).get()] = ctmp->getComp(cName).get();
      }
      tempList.push_back(cand);
    }
    prodList = tempList;
  }

private:
  std::vector<flying_ptr> fCollectSel;
};

#endif
