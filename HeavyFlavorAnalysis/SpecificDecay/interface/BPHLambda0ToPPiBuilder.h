#ifndef HeavyFlavorAnalysis_SpecificDecay_BPHLambda0ToPPiBuilder_h
#define HeavyFlavorAnalysis_SpecificDecay_BPHLambda0ToPPiBuilder_h
/** \class BPHLambda0ToPPiBuilder
 *
 *  Description: 
 *     Class to build Lambda0 to p pi candidates
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

class BPHLambda0ToPPiBuilder {
public:
  /** Constructor
   */
  BPHLambda0ToPPiBuilder(const edm::EventSetup& es,
                         const BPHRecoBuilder::BPHGenericCollection* protonCollection,
                         const BPHRecoBuilder::BPHGenericCollection* pionCollection);
  //new
  BPHLambda0ToPPiBuilder(const edm::EventSetup& es,
                         const std::vector<reco::VertexCompositeCandidate>* v0Collection,
                         const std::string& searchList = "cfp");
  BPHLambda0ToPPiBuilder(const edm::EventSetup& es,
                         const std::vector<reco::VertexCompositePtrCandidate>* vpCollection,
                         const std::string& searchList = "cfp");

  // deleted copy constructor and assignment operator
  BPHLambda0ToPPiBuilder(const BPHLambda0ToPPiBuilder& x) = delete;
  BPHLambda0ToPPiBuilder& operator=(const BPHLambda0ToPPiBuilder& x) = delete;

  /** Destructor
   */
  virtual ~BPHLambda0ToPPiBuilder();

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
  std::string protonName;
  std::string pionName;

  const edm::EventSetup* evSetup;
  const BPHRecoBuilder::BPHGenericCollection* prCollection;
  const BPHRecoBuilder::BPHGenericCollection* piCollection;
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

  std::vector<BPHPlusMinusConstCandPtr> lambda0List;
};

#endif
