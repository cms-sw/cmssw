#ifndef PhysicsTools_PatUtils_interface_PATDiObjectProxy_h
#define PhysicsTools_PatUtils_interface_PATDiObjectProxy_h

#include "DataFormats/Math/interface/deltaR.h"
#include "Utilities/General/interface/ClassName.h"

#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/PatCandidates/interface/Photon.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/PatCandidates/interface/GenericParticle.h"
#include "DataFormats/PatCandidates/interface/PFParticle.h"

namespace pat {

/* Now we implement PATDiObjectProxy with typeid & static_casts on the fly. */
class DiObjectProxy {

   public: 
        /// Default constructor, requested by ROOT. NEVER use a default constructed item!
        DiObjectProxy() : cand1_(0), cand2_(0), type1_(0), type2_(0), totalP4ok_(false), totalP4_()  {}
        /// Constructor of the pair from two Candidates
        /// Note: the Proxy MUST NOT outlive the Candidates, otherwise you get dangling pointers
        DiObjectProxy(const reco::Candidate &c1, const reco::Candidate &c2) :
            cand1_(&c1), cand2_(&c2), type1_(&typeid(c1)), type2_(&typeid(c2)), totalP4ok_(false), totalP4_() {}
        
        /// Gets the first Candidate
        const reco::Candidate & cand1() const { return *cand1_; }
        /// Gets the second Candidate
        const reco::Candidate & cand2() const { return *cand2_; }

        /// Get the angular separation
        double deltaR() const { return ::deltaR(*cand1_, *cand2_); }
        /// Get the phi separation
        double deltaPhi() const { return ::deltaPhi(cand1_->phi(), cand2_->phi()); }

        /// Get the total four momentum
        // Implementation notice: return by const reference, not by value, 
        // as it's easier for Reflex.
        const reco::Candidate::LorentzVector & totalP4() const { 
            if (!totalP4ok_) { 
                totalP4_ = cand1_->p4() + cand2_->p4(); 
                totalP4ok_ = true; 
            }
            return totalP4_;
        }

        /// Get the PAT Electron, if the pair contains one and only one PAT Electron (throw exception otherwise)
        const Electron & ele()  const { return tryGetOne_<Electron>(); }
        /// Get the PAT Muon, if the pair contains one and only one PAT Muon (throw exception otherwise)
        const Muon & mu()  const { return tryGetOne_<Muon>(); }
        /// Get the PAT Tau, if the pair contains one and only one PAT Tau (throw exception otherwise)
        const Tau & tau()  const { return tryGetOne_<Tau>(); }
        /// Get the PAT Photon, if the pair contains one and only one PAT Photon (throw exception otherwise)
        const Photon & gam()  const { return tryGetOne_<Photon>(); }
        /// Get the PAT Jet, if the pair contains one and only one PAT Jet (throw exception otherwise)
        const Jet & jet()  const { return tryGetOne_<Jet>(); }
        /// Get the PAT MET, if the pair contains one and only one PAT MET (throw exception otherwise)
        const MET & met()  const { return tryGetOne_<MET>(); }
        /// Get the PAT GenericParticle, if the pair contains one and only one PAT GenericParticle (throw exception otherwise)
        const GenericParticle & part()  const { return tryGetOne_<GenericParticle>(); }
        /// Get the PAT PFParticle, if the pair contains one and only one PAT PFParticle (throw exception otherwise)
        const PFParticle & pf()  const { return tryGetOne_<PFParticle>(); }

        /// Get the first item, if it's a PAT Electron (throw exception otherwise)
        const Electron & ele1() const { return tryGet_<Electron>(cand1_, type1_); }
        /// Get the first item, if it's a PAT Muon (throw exception otherwise)
        const Muon & mu1()  const { return tryGet_<Muon>(cand1_, type1_); }
        /// Get the first item, if it's a PAT Tau (throw exception otherwise)
        const Tau & tau1()  const { return tryGet_<Tau>(cand1_, type1_); }
        /// Get the first item, if it's a PAT Photon (throw exception otherwise)
        const Photon & gam1()  const { return tryGet_<Photon>(cand1_, type1_); }
        /// Get the first item, if it's a PAT Jet (throw exception otherwise)
        const Jet & jet1()  const { return tryGet_<Jet>(cand1_, type1_); }
        /// Get the first item, if it's a PAT MET (throw exception otherwise)
        const MET & met1()  const { return tryGet_<MET>(cand1_, type1_); }
        /// Get the first item, if it's a PAT GenericParticle (throw exception otherwise)
        const GenericParticle & part1()  const { return tryGet_<GenericParticle>(cand1_, type1_); }
        /// Get the first item, if it's a PAT PFParticle (throw exception otherwise)
        const PFParticle & pf1()  const { return tryGet_<PFParticle>(cand1_, type1_); }

        /// Get the second item, if it's a PAT Electron (throw exception otherwise)
        const Electron & ele2() const { return tryGet_<Electron>(cand2_, type2_); }
        /// Get the second item, if it's a PAT Muon (throw exception otherwise)
        const Muon & mu2()  const { return tryGet_<Muon>(cand2_, type2_); }
        /// Get the second item, if it's a PAT Tau (throw exception otherwise)
        const Tau & tau2()  const { return tryGet_<Tau>(cand2_, type2_); }
        /// Get the second item, if it's a PAT Photon (throw exception otherwise)
        const Photon & gam2()  const { return tryGet_<Photon>(cand2_, type2_); }
        /// Get the second item, if it's a PAT Jet (throw exception otherwise)
        const Jet & jet2()  const { return tryGet_<Jet>(cand2_, type2_); }
        /// Get the second item, if it's a PAT MET (throw exception otherwise)
        const MET & met2()  const { return tryGet_<MET>(cand2_, type2_); }
        /// Get the second item, if it's a PAT GenericParticle (throw exception otherwise)
        const GenericParticle & part2()  const { return tryGet_<GenericParticle>(cand2_, type2_); }
        /// Get the second item, if it's a PAT PFParticle (throw exception otherwise)
        const PFParticle & pf2()  const { return tryGet_<PFParticle>(cand2_, type2_); }

    private:

        template<typename T>
        const T & tryGet_(const reco::Candidate *ptr, const std::type_info *type) const {
            if (typeid(T) != *type) {
                throw cms::Exception("Type Error") << "pat::DiObjectProxy: the object of the pair is not of the type you request.\n" 
                                                   << " Item Index in pair: " << (ptr == cand1_ ? "first" : "second") << "\n"
                                                   << " Requested TypeID  : " << ClassName<T>::name() << "\n"
                                                   << " Found TypeID      : " << className(*ptr) << "\n";
            }
            return static_cast<const T &>(*ptr);
        }

        template<typename T> 
        const T & tryGetOne_() const {
            if (typeid(T) == *type1_) {
                if (typeid(T) == *type2_) {
                    throw cms::Exception("Type Error") << "pat::DiObjectProxy: " << 
                        "you can't get use methods that get a particle by type if the two are of the same type!\n" <<
                        " Requested Type:" << ClassName<T>::name() << "\n";
                }
                return static_cast<const T &>(*cand1_);
            } else {
                if (typeid(T) != *type2_) {
                    throw cms::Exception("Type Error") << "pat::DiObjectProxy: " << 
                        "you can't get use methods that get a particle by type if neither of the two is of that type!\n" <<
                        " Requested Type:" << ClassName<T>::name() << "\n" <<
                        " Type of first :" << className(*cand1_) << "\n" <<
                        " Type of second:" << className(*cand2_) << "\n";
                }
                return static_cast<const T &>(*cand2_);
            }
        }

       const reco::Candidate  *cand1_, *cand2_;
       const std::type_info   *type1_, *type2_;

       mutable bool totalP4ok_;
       mutable reco::Candidate::LorentzVector totalP4_;
       

};

}
#endif
