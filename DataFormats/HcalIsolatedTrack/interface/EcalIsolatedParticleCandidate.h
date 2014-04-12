#ifndef HcalIsolatedTrack_EcalIsolatedParticleCandidate_h
#define HcalIsolatedTrack_EcalIsolatedParticleCandidate_h
/** \class reco::EcalIsolatedParticleCandidate
 *
 *
 */
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/Candidate/interface/OverlapChecker.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/HcalIsolatedTrack/interface/EcalIsolatedParticleCandidateFwd.h"

namespace reco {
  
  class EcalIsolatedParticleCandidate: public LeafCandidate {
    
  public:
    
    // default constructor
    EcalIsolatedParticleCandidate() : LeafCandidate() { }
      // constructor from a tau jet
    EcalIsolatedParticleCandidate(const l1extra::L1JetParticleRef& l1tau, double etatau, double phitau,  double enIn, double enOut, int nhitIn, int nhitOut): 
      LeafCandidate( 0, PtEtaPhiMass(0.,etatau, phitau, 0.) )
      ,l1tau_(l1tau), enIn_(enIn), enOut_(enOut), nhitIn_(nhitIn), nhitOut_(nhitOut){}
    
    //constructor with null candidate
    EcalIsolatedParticleCandidate(double etatau, double phitau,  double enIn, double enOut, int nhitIn, int nhitOut):
      LeafCandidate( 0, PtEtaPhiMass(0.,etatau, phitau, 0.) )
      ,enIn_(enIn), enOut_(enOut), nhitIn_(nhitIn), nhitOut_(nhitOut) {} 

    /// destructor
    virtual ~EcalIsolatedParticleCandidate();
    /// returns a clone of the candidate
    virtual EcalIsolatedParticleCandidate * clone() const;
    
    /// reference to a tau jet
    virtual l1extra::L1JetParticleRef l1TauJet() const;
    
    /// ECAL energy in the inner cone around tau jet
    double energyIn() const {return enIn_; }
    
    /// ECAL energy in the outer cone around tau jet
    double energyOut() const {return enOut_;}
    
    /// number of ECAL hits in the inner cone around tau jet
    int nHitIn() const {return nhitIn_;}
    
    /// number of ECAL hits in the outer cone around tau jet
    int nHitOut() const {return nhitOut_;}
    
    /// set reference to l1 tau jet
    void setL1TauJet( const l1extra::L1JetParticleRef & l1tau ) { l1tau_ = l1tau; }
    
    
  private:
    /// reference to a L1 tau jet
    l1extra::L1JetParticleRef l1tau_;
    /// energy in inner cone around L1 tau jet
    double enIn_;
    /// energy in outer cone around L1 tau jet
    double enOut_;
    /// number of hits in inner cone
    int nhitIn_;
    /// number of hits in inner cone
    int nhitOut_;

  };


}

#endif // HcalIsolatedTrack_EcalIsolatedParticleCandidate_h
