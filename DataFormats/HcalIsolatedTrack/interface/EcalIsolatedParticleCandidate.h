#ifndef HcalIsolatedTrack_EcalIsolatedParticleCandidate_h
#define HcalIsolatedTrack_EcalIsolatedParticleCandidate_h
/** \class reco::EcalIsolatedParticleCandidate
 *
 *
 */

#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"

#include "DataFormats/HcalIsolatedTrack/interface/EcalIsolatedParticleCandidateFwd.h"

namespace reco {
  
  class EcalIsolatedParticleCandidate: public RecoCandidate {
    
  public:
    
    // default constructor
    EcalIsolatedParticleCandidate() : RecoCandidate() { }
      // constructor from a superCluster
      EcalIsolatedParticleCandidate(const reco::SuperClusterRef& sc,double etacl, double phicl,  double encl, double enmnear, double ensnear): 
	//     EcalIsolatedParticleCandidate(double etacl, double phicl,  double encl, double eSC, double eBC):
	RecoCandidate( 0, LorentzVector() ),
	superClu_(sc), eta_(etacl),phi_(phicl),energy_(encl), enMaxNear_(enmnear), enSumNear_(ensnear) {}
	
	//constructor with null candidate
	EcalIsolatedParticleCandidate(double etacl, double phicl,  double encl, double enmnear, double ensnear):
	  RecoCandidate( 0, LorentzVector() ), eta_(etacl),phi_(phicl),energy_(encl), enMaxNear_(enmnear), enSumNear_(ensnear) {} 
	  /// destructor
	virtual ~EcalIsolatedParticleCandidate();
	/// returns a clone of the candidate
	virtual EcalIsolatedParticleCandidate * clone() const;
	
	/// reference to a BasicCluster
	virtual reco::SuperClusterRef superCluster() const;

	double eta() const {return eta_; }
	
	double phi() const {return phi_; }

	double energy() const {return energy_; }

	/// total ecal energy in smaller cone around the candidate
	double enMaxNear() const {return enMaxNear_;}
	/// total ecal energy in bigger cone around the candidate
	double enSumNear() const {return enSumNear_;}
	
	/// set refrence to BasicCluster component
	void setSuperCluster( const SuperClusterRef & sc ) { superClu_ = sc; }
	

  private:
    /// check overlap with another candidate
    virtual bool overlap( const reco::Candidate & ) const;
    /// reference to a superCluster
    reco::SuperClusterRef superClu_;
    /// eta of super cluster
    double eta_;
    /// phi of super cluster
    double phi_;
    /// energy of super cluster 
    double energy_;
    /// total ecal energy in smaller cone around the candidate
    double enMaxNear_;
    /// total ecal energy in bigger cone around the candidate
    double enSumNear_;

  };


}

#endif
