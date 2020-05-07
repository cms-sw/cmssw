#ifndef HeavyFlavorAnalysis_SpecificDecay_BPHDecayToV0Builder_h
#define HeavyFlavorAnalysis_SpecificDecay_BPHDecayToV0Builder_h
/** \class BPHDecayToV0Builder
 *
 *  Description: 
 *     Class to build neutral particles decaying to a V0,
 *     starting from reco::Candidates or already reconstructed V0s
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecayGenericBuilder.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoBuilder.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoCandidate.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHPlusMinusCandidate.h"

#include "DataFormats/Candidate/interface/VertexCompositeCandidate.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHVertexCompositePtrCandidate.h"

#include "FWCore/Framework/interface/Event.h"

//---------------
// C++ Headers --
//---------------
#include <string>
#include <vector>
#include <map>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class BPHDecayToV0Builder : public BPHDecayGenericBuilder {
public:
  enum v0Type { VertexCompositeCandidate, VertexCompositePtrCandidate };
  struct V0Info {
    v0Type type;
    const void* v0;
  };

  /** Constructor
   */
  BPHDecayToV0Builder(const edm::EventSetup& es,
                      const std::string& d1Name,
                      const std::string& d2Name,
                      const BPHRecoBuilder::BPHGenericCollection* d1Collection,
                      const BPHRecoBuilder::BPHGenericCollection* d2Collection);
  BPHDecayToV0Builder(const edm::EventSetup& es,
                      const std::string& d1Name,
                      const std::string& d2Name,
                      const std::vector<reco::VertexCompositeCandidate>* v0Collection,
                      const std::string& searchList = "cfp");
  BPHDecayToV0Builder(const edm::EventSetup& es,
                      const std::string& d1Name,
                      const std::string& d2Name,
                      const std::vector<reco::VertexCompositePtrCandidate>* vpCollection,
                      const std::string& searchList = "cfp");

  // deleted copy constructor and assignment operator
  BPHDecayToV0Builder(const BPHDecayToV0Builder& x) = delete;
  BPHDecayToV0Builder& operator=(const BPHDecayToV0Builder& x) = delete;

  /** Destructor
   */
  ~BPHDecayToV0Builder() override;

  /** Operations
   */
  /// build candidates
  std::vector<BPHPlusMinusConstCandPtr> build();

  /// set cuts
  void setPtMin(double pt);
  void setEtaMax(double eta);

  /// get current cuts
  double getPtMin() const;
  double getEtaMax() const;

protected:
  std::vector<BPHPlusMinusConstCandPtr> cList;

  std::string p1Name;
  std::string p2Name;

  const BPHRecoBuilder::BPHGenericCollection* p1Collection;
  const BPHRecoBuilder::BPHGenericCollection* p2Collection;
  const std::vector<reco::VertexCompositeCandidate>* vCollection;
  const std::vector<reco::VertexCompositePtrCandidate>* rCollection;
  std::string sList;

  double ptMin;
  double etaMax;

  std::map<const BPHRecoCandidate*, const V0Info*> v0Map;

  /// build candidates and link to V0
  virtual void buildFromBPHGenericCollection() = 0;
  template <class T>
  void buildFromV0(const T* v0Collection, v0Type type);
  virtual BPHPlusMinusCandidatePtr buildCandidate(const reco::Candidate* c1,
                                                  const reco::Candidate* c2,
                                                  const void* v0,
                                                  v0Type type) = 0;
  void v0Clear();
};

#endif
