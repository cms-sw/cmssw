#ifndef HeavyFlavorAnalysis_RecoDecay_BPHKinematicFit_h
#define HeavyFlavorAnalysis_RecoDecay_BPHKinematicFit_h
/** \class BPHKinematicFit
 *
 *  Description: 
 *     Highest-level base class to encapsulate kinematic fit operations
 *
 *  \author Paolo Ronchese INFN Padova
 *     high-level base class to perform a kinematic fit
 */

//----------------------
// Base Class Headers --
//----------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHDecayVertex.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "RecoVertex/KinematicFitPrimitives/interface/RefCountedKinematicParticle.h"
#include "RecoVertex/KinematicFitPrimitives/interface/RefCountedKinematicTree.h"

class MultiTrackKinematicConstraint;
class KinematicConstraint;

//---------------
// C++ Headers --
//---------------
#include <string>
#include <vector>
#include <map>
#include <set>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class BPHKinematicFit : public virtual BPHDecayVertex {
public:
  /** Constructors are protected
   *  this object can exist only as part of a derived class
   */
  // deleted copy constructor and assignment operator
  BPHKinematicFit(const BPHKinematicFit& x) = delete;
  BPHKinematicFit& operator=(const BPHKinematicFit& x) = delete;

  /** Destructor
   */
  ~BPHKinematicFit() override = default;

  /** Operations
   */

  /// apply a mass constraint
  void setConstraint(double mass, double sigma);
  /// retrieve the constraint
  double constrMass() const;
  double constrSigma() const;
  /// set a decaying daughter as an unique particle fitted independently
  void setIndependentFit(const std::string& name, bool flag = true, double mass = -1.0, double sigma = -1.0);

  /// get kinematic particles
  virtual const std::vector<RefCountedKinematicParticle>& kinParticles() const;
  virtual std::vector<RefCountedKinematicParticle> kinParticles(const std::vector<std::string>& names) const;

  /// perform the kinematic fit and get the result
  virtual const RefCountedKinematicTree& kinematicTree() const;
  virtual const RefCountedKinematicTree& kinematicTree(const std::string& name, double mass, double sigma) const;
  virtual const RefCountedKinematicTree& kinematicTree(const std::string& name, double mass) const;
  virtual const RefCountedKinematicTree& kinematicTree(const std::string& name) const;
  virtual const RefCountedKinematicTree& kinematicTree(const std::string& name, KinematicConstraint* kc) const;
  virtual const RefCountedKinematicTree& kinematicTree(const std::string& name,
                                                       MultiTrackKinematicConstraint* kc) const;

  /// reset the kinematic fit
  virtual void resetKinematicFit() const;

  /// get fit status
  virtual bool isEmpty() const;
  virtual bool isValidFit() const;

  /// get current particle
  virtual const RefCountedKinematicParticle currentParticle() const;
  virtual const RefCountedKinematicVertex currentDecayVertex() const;

  /// get top particle
  virtual const RefCountedKinematicParticle topParticle() const;
  virtual const RefCountedKinematicVertex topDecayVertex() const;
  virtual ParticleMass mass() const;

  /// compute total momentum after the fit
  virtual const math::XYZTLorentzVector& p4() const;

  /// retrieve particle mass sigma
  double getMassSigma(const reco::Candidate* cand) const;

  /// retrieve independent fit flag
  bool getIndependentFit(const std::string& name, double& mass, double& sigma) const;

protected:
  // constructors
  BPHKinematicFit(int daugNum = 2, int compNum = 2);
  // pointer used to retrieve informations from other bases
  BPHKinematicFit(const BPHKinematicFit* ptr);

  // add a simple particle giving it a name
  // particles are cloned, eventually specifying a different mass
  // and a sigma
  virtual void addK(const std::string& name, const reco::Candidate* daug, double mass = -1.0, double sigma = -1.0);
  // add a simple particle and specify a criterion to search for
  // the associated track
  virtual void addK(const std::string& name,
                    const reco::Candidate* daug,
                    const std::string& searchList,
                    double mass = -1.0,
                    double sigma = -1.0);
  // add a previously reconstructed particle giving it a name
  virtual void addK(const std::string& name, const BPHRecoConstCandPtr& comp);

  // utility function used to cash reconstruction results
  void setNotUpdated() const override;

private:
  // mass constraint
  double massConst;
  double massSigma;

  // map linking daughters to mass sigma
  std::map<const reco::Candidate*, double> dMSig;

  // map to handle composite daughters as single particles
  struct FlyingParticle {
    bool flag = false;
    double mass = -1.0;
    double sigma = -1.0;
  };
  std::map<const BPHRecoCandidate*, FlyingParticle> cKinP;

  // temporary particle set
  mutable std::vector<BPHRecoConstCandPtr> tmpList;

  // reconstruction results cache
  mutable bool oldKPs;
  mutable bool oldFit;
  mutable bool oldMom;
  mutable std::map<const reco::Candidate*, RefCountedKinematicParticle> kinMap;
  mutable std::map<const BPHRecoCandidate*, RefCountedKinematicParticle> kCDMap;
  mutable std::vector<RefCountedKinematicParticle> allParticles;
  mutable RefCountedKinematicTree kinTree;
  mutable math::XYZTLorentzVector totalMomentum;

  // build kin particles, perform the fit and compute the total momentum
  virtual void buildParticles() const;
  virtual void addParticles(std::vector<RefCountedKinematicParticle>& kl,
                            std::map<const reco::Candidate*, RefCountedKinematicParticle>& km,
                            std::map<const BPHRecoCandidate*, RefCountedKinematicParticle>& cm) const;
  virtual void getParticles(const std::string& moth,
                            const std::string& daug,
                            std::vector<RefCountedKinematicParticle>& kl,
                            std::set<RefCountedKinematicParticle>& ks,
                            const BPHKinematicFit* curr) const;
  virtual void getParticles(const std::string& moth,
                            const std::vector<std::string>& daug,
                            std::vector<RefCountedKinematicParticle>& kl,
                            std::set<RefCountedKinematicParticle>& ks,
                            const BPHKinematicFit* curr) const;
  virtual unsigned int numParticles(const BPHKinematicFit* cand = nullptr) const;
  static void insertParticle(RefCountedKinematicParticle& kp,
                             std::vector<RefCountedKinematicParticle>& kl,
                             std::set<RefCountedKinematicParticle>& ks);
  virtual const BPHKinematicFit* splitKP(const std::string& name,
                                         std::vector<RefCountedKinematicParticle>* kComp,
                                         std::vector<RefCountedKinematicParticle>* kTail = nullptr) const;
  virtual const RefCountedKinematicTree& kinematicTree(const std::vector<RefCountedKinematicParticle>& kPart,
                                                       MultiTrackKinematicConstraint* kc) const;
  virtual void fitMomentum() const;
};

#endif
