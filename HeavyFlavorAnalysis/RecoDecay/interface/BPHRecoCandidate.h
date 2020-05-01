#ifndef HeavyFlavorAnalysis_RecoDecay_BPHRecoCandidate_h
#define HeavyFlavorAnalysis_RecoDecay_BPHRecoCandidate_h
/** \class BPHRecoCandidate
 *
 *  Description: 
 *     High level class for reconstructed decay candidates:
 *     - only constructor interfaces are defined in this class
 *     - functions returning results are defined in base classes:
 *       BPHDecayMomentum : decay products simple momentum sum
 *       BPHDecayVertex   : vertex reconstruction
 *       BPHKinematicFit  : kinematic fit and fitted momentum sum
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHKinematicFit.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoBuilder.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHGenericPtr.h"
#include "DataFormats/PatCandidates/interface/CompositeCandidate.h"

namespace edm {
  class EventSetup;
}

namespace reco {
  class Candidate;
}

//---------------
// C++ Headers --
//---------------
#include <vector>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class BPHRecoCandidate : public virtual BPHKinematicFit {
public:
  /** Constructor
   */
  /// create an "empty" object to add daughters later
  /// (see BPHDecayMomentum)
  BPHRecoCandidate(const edm::EventSetup* es);
  // create an object with daughters as specified in the ComponentSet
  BPHRecoCandidate(const edm::EventSetup* es, const BPHRecoBuilder::ComponentSet& compSet);

  // deleted copy constructor and assignment operator
  BPHRecoCandidate(const BPHRecoCandidate& x) = delete;
  BPHRecoCandidate& operator=(const BPHRecoCandidate& x) = delete;

  /** Destructor
   */
  ~BPHRecoCandidate() override;

  /** Operations
   */
  /// add a simple particle giving it a name
  /// particles are cloned, eventually specifying a different mass
  /// and a sigma
  virtual void add(const std::string& name, const reco::Candidate* daug, double mass = -1.0, double sigma = -1.0) {
    addK(name, daug, "cfhpmig", mass, sigma);
    return;
  }
  virtual void add(const std::string& name,
                   const reco::Candidate* daug,
                   const std::string& searchList,
                   double mass = -1.0,
                   double sigma = -1.0) {
    addK(name, daug, searchList, mass, sigma);
    return;
  }
  virtual void add(const std::string& name, const BPHRecoConstCandPtr& comp) {
    addK(name, comp);
    return;
  }

  /// look for candidates starting from particle collections as
  /// specified in the BPHRecoBuilder
  static std::vector<BPHRecoConstCandPtr> build(const BPHRecoBuilder& builder, double mass = -1, double msig = -1);

  /// clone object, cloning daughters as well up to required depth
  /// level = -1 to clone all levels
  virtual BPHRecoCandidate* clone(int level = -1) const;

protected:
  // function doing the job to clone reconstructed decays:
  // copy stable particles and clone cascade decays up to chosen level
  void fill(BPHRecoCandidate* ptr, int level) const override;

  // template function called by "build" to allow
  // the creation of derived objects
  template <class T>
  static void fill(std::vector<typename BPHGenericPtr<const T>::type>& cList,
                   const BPHRecoBuilder& builder,
                   double mass = -1,
                   double msig = -1);
};

template <class T>
void BPHRecoCandidate::fill(std::vector<typename BPHGenericPtr<const T>::type>& cList,
                            const BPHRecoBuilder& builder,
                            double mass,
                            double msig) {
  // create particle combinations
  const std::vector<BPHRecoBuilder::ComponentSet> dll = builder.build();
  // loop over combinations and create reconstructed particles
  int i;
  int n = dll.size();
  cList.reserve(n);
  T* rc = nullptr;
  for (i = 0; i < n; ++i) {
    // create reconstructed particle
    rc = new T(builder.eventSetup(), dll[i]);
    // apply mass constraint, if requested
    if (mass > 0)
      rc->setConstraint(mass, msig);
    // apply post selection
    if (builder.accept(*rc))
      cList.push_back(typename BPHGenericPtr<const T>::type(rc));
    else
      delete rc;
  }
  return;
}

#endif
