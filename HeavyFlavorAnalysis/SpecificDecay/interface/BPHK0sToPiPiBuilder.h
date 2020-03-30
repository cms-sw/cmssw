#ifndef HeavyFlavorAnalysis_SpecificDecay_BPHK0sToPiPiBuilder_h
#define HeavyFlavorAnalysis_SpecificDecay_BPHK0sToPiPiBuilder_h
/** \class BPHK0sToPiPiBuilder
 *
 *  Description: 
 *     Class to build K0s to pi+ pi- candidates
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoBuilder.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoCandidate.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHPlusMinusCandidate.h"

#include "DataFormats/Candidate/interface/VertexCompositeCandidate.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHVertexCompositePtrCandidate.h"

#include "FWCore/Framework/interface/Event.h"

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

class BPHK0sToPiPiBuilder {
public:
  /** Constructor
   */
  BPHK0sToPiPiBuilder(const edm::EventSetup& es,
                      const BPHRecoBuilder::BPHGenericCollection* posCollection,
                      const BPHRecoBuilder::BPHGenericCollection* negCollection);
  BPHK0sToPiPiBuilder(const edm::EventSetup& es,
                      const std::vector<reco::VertexCompositeCandidate>* v0Collection,
                      const std::string& searchList = "cfp");
  BPHK0sToPiPiBuilder(const edm::EventSetup& es,
                      const std::vector<reco::VertexCompositePtrCandidate>* vpCollection,
                      const std::string& searchList = "cfp");

  // deleted copy constructor and assignment operator
  BPHK0sToPiPiBuilder(const BPHK0sToPiPiBuilder& x) = delete;
  BPHK0sToPiPiBuilder& operator=(const BPHK0sToPiPiBuilder& x) = delete;

  /** Destructor
   */
  virtual ~BPHK0sToPiPiBuilder();

  /** Operations
   */
  /// build Phi candidates
  std::vector<BPHPlusMinusConstCandPtr> build();

  /// set cuts
  void setPtMin(double pt);
  void setEtaMax(double eta);
  void setMassMin(double m);
  void setMassMax(double m);
  void setProbMin(double p);
  void setConstr(double mass, double sigma);

  /// get current cuts
  double getPtMin() const;
  double getEtaMax() const;
  double getMassMin() const;
  double getMassMax() const;
  double getProbMin() const;
  double getConstrMass() const;
  double getConstrSigma() const;

private:
  std::string pionPosName;
  std::string pionNegName;

  const edm::EventSetup* evSetup;
  const BPHRecoBuilder::BPHGenericCollection* pCollection;
  const BPHRecoBuilder::BPHGenericCollection* nCollection;
  const std::vector<reco::VertexCompositeCandidate>* vCollection;
  const std::vector<reco::VertexCompositePtrCandidate>* rCollection;
  std::string sList;

  void buildFromBPHGenericCollection();
  template <class T>
  void buildFromV0(const T* v0Collection);

  BPHParticlePtSelect* ptSel;
  BPHParticleEtaSelect* etaSel;
  BPHMassSelect* massSel;
  BPHChi2Select* chi2Sel;
  double cMass;
  double cSigma;
  bool updated;

  std::vector<BPHPlusMinusConstCandPtr> k0sList;
};

#endif
