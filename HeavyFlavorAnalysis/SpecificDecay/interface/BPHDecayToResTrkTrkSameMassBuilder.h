#ifndef HeavyFlavorAnalysis_SpecificDecay_BPHDecayToResTrkTrkSameMassBuilder_h
#define HeavyFlavorAnalysis_SpecificDecay_BPHDecayToResTrkTrkSameMassBuilder_h
/** \class BPHDecayToResTrkTrkSameMassBuilder
 *
 *  Description: 
 *     Class to build a particle decaying to a resonance, decaying itself
 *     to an opposite charged particles pair,
 *     and two additional opposite charged particles pair
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecayConstrainedBuilder.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecayToResTrkTrkSameMassBuilderBase.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHParticlePtSelect.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHParticleEtaSelect.h"

#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoBuilder.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoCandidate.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHPlusMinusCandidate.h"

#include "FWCore/Framework/interface/EventSetup.h"

class BPHEventSetupWrapper;
class BPHParticleNeutralVeto;

//---------------
// C++ Headers --
//---------------
#include <string>
#include <vector>

//              ---------------------
//              -- Class Interface --
//              ---------------------

template <class ProdType, class ResType>
class BPHDecayToResTrkTrkSameMassBuilder : public BPHDecayToResTrkTrkSameMassBuilderBase,
                                           public BPHDecayConstrainedBuilder<ProdType, ResType> {
public:
  using typename BPHDecayGenericBuilder<ProdType>::prod_ptr;
  using typename BPHDecayConstrainedBuilder<ProdType, ResType>::res_ptr;

  /** Constructor
   */
  BPHDecayToResTrkTrkSameMassBuilder(const BPHEventSetupWrapper& es,
                                     const std::string& resName,
                                     double resMass,
                                     double resWidth,
                                     const std::vector<res_ptr>& resCollection,
                                     const std::string& posName,
                                     const std::string& negName,
                                     double trkMass,
                                     double trkSigma,
                                     const BPHRecoBuilder::BPHGenericCollection* posCollection,
                                     const BPHRecoBuilder::BPHGenericCollection* negCollection)
      : BPHDecayGenericBuilderBase(es, nullptr),
        BPHDecayConstrainedBuilderBase(resName, resMass, resWidth),
        BPHDecayToResTrkTrkSameMassBuilderBase(posName, negName, trkMass, trkSigma, posCollection, negCollection),
        BPHDecayConstrainedBuilder<ProdType, ResType>(resCollection) {}

  // deleted copy constructor and assignment operator
  BPHDecayToResTrkTrkSameMassBuilder(const BPHDecayToResTrkTrkSameMassBuilder& x) = delete;
  BPHDecayToResTrkTrkSameMassBuilder& operator=(const BPHDecayToResTrkTrkSameMassBuilder& x) = delete;

  /** Destructor
   */
  ~BPHDecayToResTrkTrkSameMassBuilder() override = default;

protected:
  BPHDecayToResTrkTrkSameMassBuilder(const std::vector<res_ptr>& resCollection,
                                     const std::string& posName,
                                     const std::string& negName,
                                     double trkMass,
                                     double trkSigma,
                                     const BPHRecoBuilder::BPHGenericCollection* posCollection,
                                     const BPHRecoBuilder::BPHGenericCollection* negCollection)
      : BPHDecayToResTrkTrkSameMassBuilderBase(posName, negName, trkMass, trkSigma, posCollection, negCollection),
        BPHDecayConstrainedBuilder<ProdType, ResType>(resCollection) {}

  void fillRecList() override {
    std::vector<res_ptr> resList;
    int nRes = this->rCollection->size();
    int iRes;
    resList.reserve(nRes);
    for (iRes = 0; iRes < nRes; ++iRes) {
      const res_ptr& rCand = this->rCollection->at(iRes);
      if (this->resoSel->accept(*rCand))
        resList.push_back(rCand);
    }
    if (resList.empty())
      return;
    nRes = resList.size();

    fillTrkTrkList();
    if (ttPairs.empty())
      return;

    int nPair = ttPairs.size();
    int iPair;
    for (iPair = 0; iPair < nPair; ++iPair) {
      const BPHPlusMinusConstCandPtr tt = ttPairs[iPair];
      for (iRes = 0; iRes < nRes; ++iRes) {
        ProdType* cand = new ProdType(evSetup);
        prod_ptr cPtr(cand);
        cand->add(rName, resList[iRes]);
        cand->add(pName, tt->originalReco(tt->getDaug(pName)), tMass, tSigma);
        cand->add(nName, tt->originalReco(tt->getDaug(nName)), tMass, tSigma);
        if (!massSel->accept(*cand))
          continue;
        if ((chi2Sel != nullptr) && !chi2Sel->accept(*cand))
          continue;
        if (!mFitSel->accept(*cand))
          continue;
        this->recList.push_back(cPtr);
      }
    }
    ttPairs.clear();

    return;
  }
};

#endif
