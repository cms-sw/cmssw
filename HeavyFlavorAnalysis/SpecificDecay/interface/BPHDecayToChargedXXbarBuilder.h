#ifndef HeavyFlavorAnalysis_SpecificDecay_BPHDecayToChargedXXbarBuilder_h
#define HeavyFlavorAnalysis_SpecificDecay_BPHDecayToChargedXXbarBuilder_h
/** \class BPHDecayToChargedXXbarBuilder
 *
 *  Description: 
 *     Class to build a decay to an oppositely charged 
 *     particle-antiparticle pair
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

class BPHDecayToChargedXXbarBuilder : public virtual BPHDecayGenericBuilderBase,
                                      public virtual BPHDecayGenericBuilder<BPHPlusMinusCandidate> {
public:
  /** Constructor
   */
  BPHDecayToChargedXXbarBuilder(const BPHEventSetupWrapper& es,
                                const std::string& dPosName,
                                const std::string& dNegName,
                                double daugMass,
                                double daugSigma,
                                const BPHRecoBuilder::BPHGenericCollection* posCollection,
                                const BPHRecoBuilder::BPHGenericCollection* negCollection);

  // deleted copy constructor and assignment operator
  BPHDecayToChargedXXbarBuilder(const BPHDecayToChargedXXbarBuilder& x) = delete;
  BPHDecayToChargedXXbarBuilder& operator=(const BPHDecayToChargedXXbarBuilder& x) = delete;

  /** Destructor
   */
  ~BPHDecayToChargedXXbarBuilder() override = default;

  /** Operations
   */

  /// set cuts
  void setPtMin(double pt);
  void setEtaMax(double eta);
  void setDzMax(double dz);

  /// get current cuts
  double getPtMin() const { return ptMin; }
  double getEtaMax() const { return etaMax; }
  double getDzMax() const { return dzMax; }

protected:
  double ptMin;
  double etaMax;
  double dzMax;

  /// build candidates
  void fillRecList() override;

protected:
  std::string pName;
  std::string nName;
  double dMass;
  double dSigma;

  const BPHRecoBuilder::BPHGenericCollection* pCollection;
  const BPHRecoBuilder::BPHGenericCollection* nCollection;

private:
  class Particle {
  public:
    Particle(const reco::Candidate* c, const reco::Track* tk, double x, double y, double z, double e)
        : cand(c), track(tk), px(x), py(y), pz(z), en(e) {}
    const reco::Candidate* cand;
    const reco::Track* track;
    double px;
    double py;
    double pz;
    double en;
  };
  void addParticle(const BPHRecoBuilder::BPHGenericCollection* collection, int charge, std::vector<Particle*>& list);
};

#endif
