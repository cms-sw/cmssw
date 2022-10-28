#ifndef HeavyFlavorAnalysis_SpecificDecay_BPHDecayToTkpTknSymChargeBuilder_h
#define HeavyFlavorAnalysis_SpecificDecay_BPHDecayToTkpTknSymChargeBuilder_h
/** \class BPHDecayToTkpTknSymChargeBuilder
 *
 *  Description: 
 *     Class to build a decay to an oppositely charged particle pair
 *     with different masses, choosing the best mass assignment on
 *     the reconstructed mass basis
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecayGenericBuilderBase.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecayGenericBuilder.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoBuilder.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoCandidate.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHPlusMinusCandidate.h"

class BPHEventSetupWrapper;
class BPHParticlePtSelect;
class BPHParticleEtaSelect;
class BPHChi2Select;
class BPHMassSelect;

//---------------
// C++ Headers --
//---------------
#include <string>
#include <vector>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class BPHDecayToTkpTknSymChargeBuilder : public virtual BPHDecayGenericBuilderBase,
                                         public virtual BPHDecayGenericBuilder<BPHPlusMinusCandidate> {
public:
  /** Constructor
   */
  BPHDecayToTkpTknSymChargeBuilder(const BPHEventSetupWrapper& es,
                                   const std::string& daug1Name,
                                   double daug1Mass,
                                   double daug1Sigma,
                                   const std::string& daug2Name,
                                   double daug2Mass,
                                   double daug2Sigma,
                                   const BPHRecoBuilder::BPHGenericCollection* posCollection,
                                   const BPHRecoBuilder::BPHGenericCollection* negCollection,
                                   double expectedMass);

  // deleted copy constructor and assignment operator
  BPHDecayToTkpTknSymChargeBuilder(const BPHDecayToTkpTknSymChargeBuilder& x) = delete;
  BPHDecayToTkpTknSymChargeBuilder& operator=(const BPHDecayToTkpTknSymChargeBuilder& x) = delete;

  /** Destructor
   */
  ~BPHDecayToTkpTknSymChargeBuilder() override = default;

  /** Operations
   */

  /// set cuts
  void setTrk1PtMin(double pt);
  void setTrk2PtMin(double pt);
  void setTrk1EtaMax(double eta);
  void setTrk2EtaMax(double eta);
  void setDzMax(double dz);

  /// get current cuts
  double getTrk1PtMin() const { return pt1Min; }
  double getTrk2PtMin() const { return pt2Min; }
  double getTrk1EtaMax() const { return eta1Max; }
  double getTrk2EtaMax() const { return eta2Max; }
  double getDzMax() const { return dzMax; }

private:
  std::string d1Name;
  double d1Mass;
  double d1Sigma;
  std::string d2Name;
  double d2Mass;
  double d2Sigma;
  double eMass;

  const BPHRecoBuilder::BPHGenericCollection* pCollection;
  const BPHRecoBuilder::BPHGenericCollection* nCollection;

  double pt1Min;
  double pt2Min;
  double eta1Max;
  double eta2Max;
  double dzMax;

  class Particle {
  public:
    Particle(const reco::Candidate* c, const reco::Track* tk, double x, double y, double z, double f, double g)
        : cand(c), track(tk), px(x), py(y), pz(z), e1(f), e2(g) {}
    const reco::Candidate* cand;
    const reco::Track* track;
    double px;
    double py;
    double pz;
    double e1;
    double e2;
  };
  void addParticle(const BPHRecoBuilder::BPHGenericCollection* collection, int charge, std::vector<Particle*>& list);

  /// build candidates
  void fillRecList() override;
};

#endif
