#ifndef HeavyFlavorAnalysis_RecoDecay_BPHDecayMomentum_h
#define HeavyFlavorAnalysis_RecoDecay_BPHDecayMomentum_h
/** \class BPHDecayMomentum
 *
 *  Description: 
 *     Lowest-level base class to contain decay products and
 *     compute total momentum
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
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoCandidatePtr.h"
#include "DataFormats/PatCandidates/interface/CompositeCandidate.h"
class BPHRecoBuilder;

//---------------
// C++ Headers --
//---------------
#include <vector>
#include <map>
#include <string>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class BPHDecayMomentum {
public:
  /** Constructors are protected
   *  this object can exist only as part of a derived class
   */
  // deleted copy constructor and assignment operator
  BPHDecayMomentum(const BPHDecayMomentum& x) = delete;
  BPHDecayMomentum& operator=(const BPHDecayMomentum& x) = delete;

  /** Destructor
   */
  virtual ~BPHDecayMomentum();

  /** Operations
   */

  /// get a composite by the simple sum of simple particles
  virtual const pat::CompositeCandidate& composite() const;

  /// get the list of names of simple particles directly produced in the decay
  /// e.g. in JPsi -> mu+mu-   returns the two names used for muons
  ///      in B+   -> JPsi K+  returns the name used for the K+
  ///      in Bs   -> JPsi Phi returns an empty list
  virtual const std::vector<std::string>& daugNames() const;

  /// get the list of names of previously reconstructed particles
  /// e.g. in JPsi -> mu+mu-   returns an empty list
  ///      in B+   -> JPsi K+  returns the name used for the JPsi
  ///      in Bs   -> JPsi Phi returns the two names used for JPsi and Phi
  virtual const std::vector<std::string>& compNames() const;

  /// get the list of simple particles directly produced in the decay
  /// e.g. in JPsi -> mu+mu-   returns the two muons
  ///      in B+   -> JPsi K+  returns the K+
  ///      in Bs   -> JPsi Phi returns an empty list
  /// (clones are actually returned)
  virtual const std::vector<const reco::Candidate*>& daughters() const;

  /// get the full list of simple particles produced in the decay,
  /// directly or through cascade decays
  /// e.g. in B+   -> JPsi K+ ; JPsi -> mu+mu- returns the mu+, mu-, K+
  ///      in Bs   -> JPsi Phi; JPsi -> mu+mu-; Phi -> K+K-
  ///                                          returns the mu+, mu-, K+, K-
  /// (clones are actually returned)
  virtual const std::vector<const reco::Candidate*>& daughFull() const;

  /// get the original particle from the clone
  virtual const reco::Candidate* originalReco(const reco::Candidate* daug) const;

  /// get the list of previously reconstructed particles
  /// e.g. in JPsi -> mu+mu-   returns an empty list
  ///      in B+   -> JPsi K+  returns the JPsi
  ///      in Bs   -> JPsi Phi returns the JPsi and Phi
  virtual const std::vector<BPHRecoConstCandPtr>& daughComp() const;

  /// get a simple particle from the name
  /// return null pointer if not found
  virtual const reco::Candidate* getDaug(const std::string& name) const;

  /// get a previously reconstructed particle from the name
  /// return null pointer if not found
  virtual BPHRecoConstCandPtr getComp(const std::string& name) const;

  const std::map<std::string, const reco::Candidate*>& daugMap() const { return dMap; }

  const std::map<std::string, BPHRecoConstCandPtr>& compMap() const { return cMap; }

  struct Component {
    const reco::Candidate* cand;
    double mass;
    double msig;
    std::string searchList;
  };

protected:
  // constructors
  BPHDecayMomentum(int daugNum = 2, int compNum = 2);
  BPHDecayMomentum(const std::map<std::string, Component>& daugMap, int compNum = 2);
  BPHDecayMomentum(const std::map<std::string, Component>& daugMap,
                   const std::map<std::string, BPHRecoConstCandPtr> compMap);

  // get an object filled in the constructor
  // to be used in the creation of other bases of BPHRecoCandidate
  const std::vector<Component>& componentList() const;

  // add a simple particle giving it a name
  // particles are cloned, eventually specifying a different mass
  virtual void addP(const std::string& name, const reco::Candidate* daug, double mass = -1.0);
  // add a previously reconstructed particle giving it a name
  virtual void addP(const std::string& name, const BPHRecoConstCandPtr& comp);

  // utility function used to cash reconstruction results
  virtual void setNotUpdated() const;

  // function doing the job to clone reconstructed decays:
  // copy stable particles and clone cascade decays up to chosen level
  virtual void fill(BPHRecoCandidate* ptr, int level) const = 0;

private:
  // object filled in the constructor
  // to be used in the creation of other bases of BPHRecoCandidate
  std::vector<Component> compList;

  // names used for simple and previously reconstructed particles
  std::vector<std::string> nList;
  std::vector<std::string> nComp;

  // pointers to simple and previously reconstructed particles
  // (clones stored for simple particles)
  std::vector<const reco::Candidate*> dList;
  std::vector<BPHRecoConstCandPtr> cList;

  // maps linking names to decay products
  // (simple and previously reconstructed particles)
  std::map<std::string, const reco::Candidate*> dMap;
  std::map<std::string, BPHRecoConstCandPtr> cMap;

  // map linking cloned particles to original ones
  std::map<const reco::Candidate*, const reco::Candidate*> clonesMap;

  // reconstruction results cache
  mutable bool oldMom;
  mutable std::vector<const reco::Candidate*> dFull;
  mutable pat::CompositeCandidate compCand;

  // create clones of simple particles, store them and their names
  void clonesList(const std::map<std::string, Component>& daugMap);

  // fill lists of previously reconstructed particles and their names
  // and retrieve cascade decay products
  void dCompList();

  // compute the total momentum of simple particles, produced
  // directly or in cascade decays
  virtual void sumMomentum(const std::vector<const reco::Candidate*>& dl, const std::vector<std::string>& dn) const;

  // recursively fill the list of simple particles, produced
  // directly or in cascade decays
  virtual void fillDaug(std::vector<const reco::Candidate*>& ad,
                        const std::string& name,
                        std::vector<std::string>& an) const;

  // compute the total momentum and cache it
  virtual void computeMomentum() const;
};

#endif
