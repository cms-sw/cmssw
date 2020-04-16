#ifndef HeavyFlavorAnalysis_RecoDecay_BPHPlusMinusCandidate_h
#define HeavyFlavorAnalysis_RecoDecay_BPHPlusMinusCandidate_h
/** \class BPHPlusMinusCandidate
 *
 *  Description: 
 *     class for reconstructed decay candidates to opposite charge
 *     particle pairs
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoCandidate.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHPlusMinusVertex.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//---------------
// C++ Headers --
//---------------
#include <string>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class BPHPlusMinusCandidate : public BPHRecoCandidate, public virtual BPHPlusMinusVertex {
  friend class BPHRecoCandidate;

public:
  /** Constructor
   */
  BPHPlusMinusCandidate(const edm::EventSetup* es);

  // deleted copy constructor and assignment operator
  BPHPlusMinusCandidate(const BPHPlusMinusCandidate& x) = delete;
  BPHPlusMinusCandidate& operator=(const BPHPlusMinusCandidate& x) = delete;

  /** Destructor
   */
  ~BPHPlusMinusCandidate() override;

  /** Operations
   */
  /// add a simple particle giving it a name
  /// particles are cloned, eventually specifying a different mass
  /// particles can be added only up to two particles with opposite charge
  void add(const std::string& name, const reco::Candidate* daug, double mass = -1.0, double sigma = -1.0) override;
  void add(const std::string& name,
           const reco::Candidate* daug,
           const std::string& searchList,
           double mass = -1.0,
           double sigma = -1.0) override;

  /// look for candidates starting from particle collections as
  /// specified in the BPHRecoBuilder, with given names for
  /// positive and negative particle
  /// charge selection is applied inside
  static std::vector<BPHPlusMinusConstCandPtr> build(const BPHRecoBuilder& builder,
                                                     const std::string& nPos,
                                                     const std::string& nNeg,
                                                     double mass = -1,
                                                     double msig = -1);

  /// clone object, cloning daughters as well up to required depth
  /// level = -1 to clone all levels
  const BPHRecoCandidate* clone(int level = -1) const override;

  /// get a composite by the simple sum of simple particles
  const pat::CompositeCandidate& composite() const override;

  /// get cowboy/sailor classification
  bool isCowboy() const;
  bool isSailor() const;

protected:
  // utility function used to cash reconstruction results
  void setNotUpdated() const override {
    BPHKinematicFit::setNotUpdated();
    BPHPlusMinusVertex::setNotUpdated();
  }

private:
  // constructor
  BPHPlusMinusCandidate(const edm::EventSetup* es, const BPHRecoBuilder::ComponentSet& compList);

  // return true or false for positive or negative phi_pos-phi_neg difference
  bool phiDiff() const;
};

#endif
