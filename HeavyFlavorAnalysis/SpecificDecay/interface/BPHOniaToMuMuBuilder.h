#ifndef HeavyFlavorAnalysis_SpecificDecay_BPHOniaToMuMuBuilder_h
#define HeavyFlavorAnalysis_SpecificDecay_BPHOniaToMuMuBuilder_h
/** \class BPHOniaToMuMuBuilder
 *
 *  Description: 
 *     Class to build Psi(1,2) and Upsilon(1,2,3) candidates
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
class BPHMuonPtSelect;
class BPHMuonEtaSelect;
class BPHChi2Select;
class BPHMassSelect;
class BPHRecoSelect;
class BPHMomentumSelect;
class BPHVertexSelect;

//---------------
// C++ Headers --
//---------------
#include <string>
#include <vector>
#include <map>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class BPHOniaToMuMuBuilder : public virtual BPHDecayGenericBuilderBase,
                             public virtual BPHDecayGenericBuilder<BPHPlusMinusCandidate> {
public:
  enum oniaType { NRes, Phi, Psi1, Psi2, Ups, Ups1, Ups2, Ups3 };

  /** Constructor
   */
  BPHOniaToMuMuBuilder(const BPHEventSetupWrapper& es,
                       const BPHRecoBuilder::BPHGenericCollection* muPosCollection,
                       const BPHRecoBuilder::BPHGenericCollection* muNegCollection);

  // deleted copy constructor and assignment operator
  BPHOniaToMuMuBuilder(const BPHOniaToMuMuBuilder& x) = delete;
  BPHOniaToMuMuBuilder& operator=(const BPHOniaToMuMuBuilder& x) = delete;

  /** Destructor
   */
  ~BPHOniaToMuMuBuilder() override;

  /** Operations
   */
  /// build resonance candidates
  void fillRecList() override;

  /// extract list of candidates of specific type
  /// candidates are rebuilt applying corresponding mass constraint
  std::vector<BPHPlusMinusConstCandPtr> getList(oniaType type,
                                                BPHRecoSelect* dSel = nullptr,
                                                BPHMomentumSelect* mSel = nullptr,
                                                BPHVertexSelect* vSel = nullptr,
                                                BPHFitSelect* kSel = nullptr);

  /// retrieve original candidate from a copy with the same daughters
  /// obtained through "getList"
  BPHPlusMinusConstCandPtr getOriginalCandidate(const BPHRecoCandidate& cand);

  /// set cuts
  void setPtMin(oniaType type, double pt);
  void setEtaMax(oniaType type, double eta);
  void setMassMin(oniaType type, double m);
  void setMassMax(oniaType type, double m);
  void setProbMin(oniaType type, double p);
  void setConstr(oniaType type, double mass, double sigma);

  /// get current cuts
  double getPtMin(oniaType type) const;
  double getEtaMax(oniaType type) const;
  double getMassMin(oniaType type) const;
  double getMassMax(oniaType type) const;
  double getProbMin(oniaType type) const;
  double getConstrMass(oniaType type) const;
  double getConstrSigma(oniaType type) const;

private:
  std::string muPosName;
  std::string muNegName;

  const BPHRecoBuilder::BPHGenericCollection* posCollection;
  const BPHRecoBuilder::BPHGenericCollection* negCollection;

  struct OniaParameters {
    BPHMuonPtSelect* ptSel;
    BPHMuonEtaSelect* etaSel;
    BPHMassSelect* massSel;
    BPHChi2Select* chi2Sel;
    double mass;
    double sigma;
    bool outdated;
  };

  std::map<oniaType, OniaParameters> oniaPar;
  std::map<oniaType, std::vector<BPHPlusMinusConstCandPtr> > oniaList;

  void setNotUpdated();
  void setParameters(oniaType type,
                     double ptMin,
                     double etaMax,
                     double massMin,
                     double massMax,
                     double probMin,
                     double mass,
                     double sigma);
  void extractList(oniaType type);
};

#endif
