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
#include <map>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class BPHKinematicFit: public virtual BPHDecayVertex {

 public:

  /** Constructor is protected
   *  this object can exist only as part of a derived class
   */

  /** Destructor
   */
  virtual ~BPHKinematicFit();

  /** Operations
   */

  /// apply a mass constraint
  void setConstraint( double mass, double sigma );
  /// retrieve the constraint
  double constrMass() const;
  double constrSigma() const;

  /// get kinematic particles
  virtual const std::vector<RefCountedKinematicParticle>& kinParticles() const;
  virtual       std::vector<RefCountedKinematicParticle>  kinParticles(
          const std::vector<std::string>& names ) const;

  /// perform the kinematic fit and get the result
  virtual const RefCountedKinematicTree& kinematicTree() const;
  virtual const RefCountedKinematicTree& kinematicTree(
          const std::string& name, double mass, double sigma ) const;
  virtual const RefCountedKinematicTree& kinematicTree(
          const std::string& name, double mass ) const;
  virtual const RefCountedKinematicTree& kinematicTree(
          const std::string& name, KinematicConstraint* kc ) const;
  virtual const RefCountedKinematicTree& kinematicTree(
          const std::string& name, MultiTrackKinematicConstraint* kc ) const;

  /// reset the kinematic fit
  virtual void resetKinematicFit() const;

  // get current particle
  virtual bool isEmpty() const;
  virtual bool isValidFit() const;
  virtual const RefCountedKinematicParticle currentParticle   () const;
  virtual const RefCountedKinematicVertex   currentDecayVertex() const;
  virtual ParticleMass                      mass              () const;

  /// compute total momentum after the fit
  virtual const math::XYZTLorentzVector& p4() const;

 protected:

  // constructors
  BPHKinematicFit();
  // pointer used to retrieve informations from other bases
  BPHKinematicFit( const BPHKinematicFit* ptr );

  /// add a simple particle giving it a name
  /// particles are cloned, eventually specifying a different mass
  /// and a sigma
  virtual void addK( const std::string& name,
                     const reco::Candidate* daug,
                     double mass = -1.0, double sigma = -1.0 );
  /// add a simple particle and specify a criterion to search for
  /// the associated track
  virtual void addK( const std::string& name,
                     const reco::Candidate* daug,
                     const std::string& searchList,
                     double mass = -1.0, double sigma = -1.0 );
  /// add a previously reconstructed particle giving it a name
  virtual void addK( const std::string& name,
                     const BPHRecoConstCandPtr& comp );

  // utility function used to cash reconstruction results
  virtual void setNotUpdated() const;

 private:

  // mass constraint
  double massConst;
  double massSigma;

  // map linking daughters to mass sigma
  std::map<const reco::Candidate*,double> dMSig;

  // reconstruction results cache
  mutable bool oldKPs;
  mutable bool oldFit;
  mutable bool oldMom;
  mutable std::map   <const reco::Candidate*,
                      RefCountedKinematicParticle> kinMap;
  mutable std::vector<RefCountedKinematicParticle> allParticles;
  mutable RefCountedKinematicTree kinTree;
  mutable math::XYZTLorentzVector totalMomentum;

  // build kin particles, perform the fit and compute the total momentum
  virtual void buildParticles() const;
  virtual void fitMomentum() const;

};


#endif

