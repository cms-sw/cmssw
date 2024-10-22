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
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecayGenericBuilder.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecaySpecificBuilder.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoBuilder.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoCandidate.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHPlusMinusCandidate.h"

#include "FWCore/Framework/interface/EventSetup.h"

class BPHEventSetupWrapper;
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

class BPHBdToKxMuMuBuilder : public virtual BPHDecayGenericBuilder<BPHRecoCandidate>,
                             public BPHDecaySpecificBuilder<BPHRecoCandidate> {
public:
  /** Constructor
   */
  BPHBdToKxMuMuBuilder(const BPHEventSetupWrapper& es,
                       const std::vector<BPHPlusMinusConstCandPtr>& oniaCollection,
                       const std::vector<BPHPlusMinusConstCandPtr>& kx0Collection)
      : BPHDecayGenericBuilderBase(es, nullptr),
        oniaName("MuMu"),
        kx0Name("Kx0"),
        oCollection(&oniaCollection),
        kCollection(&kx0Collection) {
    oniaSel = new BPHMassSelect(1.00, 12.00);
    mkx0Sel = new BPHMassSelect(0.80, 1.00);
  }

  // deleted copy constructor and assignment operator
  BPHBdToKxMuMuBuilder(const BPHBdToKxMuMuBuilder& x) = delete;
  BPHBdToKxMuMuBuilder& operator=(const BPHBdToKxMuMuBuilder& x) = delete;

  /** Destructor
   */
  ~BPHBdToKxMuMuBuilder() override = default;

  /** Operations
   */
  /// build candidates
  void fill(BPHRecoBuilder& brb, void* parameters) override {
    brb.setMinPDiffererence(minPDiff);
    brb.add(oniaName, *oCollection);
    brb.add(kx0Name, *kCollection);
    brb.filter(oniaName, *oniaSel);
    brb.filter(kx0Name, *mkx0Sel);
    if (massSel->getMassMax() >= 0.0)
      brb.filter(*massSel);
    if (chi2Sel->getProbMin() >= 0.0)
      brb.filter(*chi2Sel);
    return;
  }

  /// set cuts
  void setOniaMassMin(double m) {
    outdated = true;
    oniaSel->setMassMin(m);
  }
  void setOniaMassMax(double m) {
    outdated = true;
    oniaSel->setMassMax(m);
  }
  void setKxMassMin(double m) {
    outdated = true;
    mkx0Sel->setMassMin(m);
  }
  void setKxMassMax(double m) {
    outdated = true;
    mkx0Sel->setMassMax(m);
  }

  /// get current cuts
  double getOniaMassMin() const { return oniaSel->getMassMin(); }
  double getOniaMassMax() const { return oniaSel->getMassMax(); }
  double getKxMassMin() const { return mkx0Sel->getMassMin(); }
  double getKxMassMax() const { return mkx0Sel->getMassMax(); }

  /// setup parameters for BPHRecoBuilder
  void setup(void* parameters) override {}

private:
  std::string oniaName;
  std::string kx0Name;

  const std::vector<BPHPlusMinusConstCandPtr>* oCollection;
  const std::vector<BPHPlusMinusConstCandPtr>* kCollection;

  BPHMassSelect* oniaSel;
  BPHMassSelect* mkx0Sel;
};

#endif
