#ifndef HeavyFlavorAnalysis_SpecificDecay_BPHDecayToResTrkBuilderBase_h
#define HeavyFlavorAnalysis_SpecificDecay_BPHDecayToResTrkBuilderBase_h
/** \class BPHDecayToResTrkBuilderBase
 *
 *  Description: 
 *     Base class to build a particle decaying to a particle, decaying itself
 *     in cascade, and an additional track
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecaySpecificBuilder.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecayConstrainedBuilderBase.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHParticlePtSelect.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHParticleEtaSelect.h"

#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoBuilder.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoCandidate.h"

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

class BPHDecayToResTrkBuilderBase : public virtual BPHDecaySpecificBuilderBase,
                                    public virtual BPHDecayConstrainedBuilderBase {
public:
  /** Constructor
   */
  BPHDecayToResTrkBuilderBase(const BPHEventSetupWrapper& es,
                              const std::string& resName,
                              double resMass,
                              double resWidth,
                              const std::string& trkName,
                              double trkMass,
                              double trkSigma,
                              const BPHRecoBuilder::BPHGenericCollection* trkCollection);

  // deleted copy constructor and assignment operator
  BPHDecayToResTrkBuilderBase(const BPHDecayToResTrkBuilderBase& x) = delete;
  BPHDecayToResTrkBuilderBase& operator=(const BPHDecayToResTrkBuilderBase& x) = delete;

  /** Destructor
   */
  ~BPHDecayToResTrkBuilderBase() override;

  /** Operations
   */

  /// set cuts
  void setTrkPtMin(double pt);
  void setTrkEtaMax(double eta);

  /// get current cuts
  double getTrkPtMin() const { return ptSel->getPtMin(); }
  double getTrkEtaMax() const { return etaSel->getEtaMax(); }

protected:
  BPHDecayToResTrkBuilderBase(const std::string& trkName,
                              double trkMass,
                              double trkSigma,
                              const BPHRecoBuilder::BPHGenericCollection* trkCollection);

  std::string tName;
  double tMass;
  double tSigma;

  const BPHRecoBuilder::BPHGenericCollection* tCollection;

  BPHParticleNeutralVeto* tknVeto;
  BPHParticlePtSelect* ptSel;
  BPHParticleEtaSelect* etaSel;

  /// build candidates
  void fill(BPHRecoBuilder& brb, void* parameters) override;

private:
  std::vector<const reco::Candidate*> tCollectSel1;
  std::vector<const reco::Candidate*> tCollectSel2;
  static void filter(const std::vector<const reco::Candidate*>* s,
                     std::vector<const reco::Candidate*>* d,
                     BPHRecoSelect* f) {
    int i;
    int n = s->size();
    d->clear();
    d->reserve(n);
    for (i = 0; i < n; ++i) {
      if (f->accept(*s->at(i)))
        d->push_back(s->at(i));
    }
  }
  void swap(std::vector<const reco::Candidate*>*& l, std::vector<const reco::Candidate*>*& r) {
    std::vector<const reco::Candidate*>* t = l;
    l = r;
    r = t;
  }
};

#endif
