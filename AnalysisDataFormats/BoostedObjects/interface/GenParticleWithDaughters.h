#ifndef ANALYSISDATAFORMATS_BOOSTEDOBJECTS_GENPARTICLEWITHDAUGHTERS_HH
#define ANALYSISDATAFORMATS_BOOSTEDOBJECTS_GENPARTICLEWITHDAUGHTERS_HH

#include "AnalysisDataFormats/BoostedObjects/interface/GenParticle.h" 

#include <TLorentzVector.h>

namespace vlq {
  class GenParticleWithDaughters ; 
  typedef std::vector<GenParticleWithDaughters> GenParticleWithDaughtersCollection ;  
  class GenParticleWithDaughters {
    friend class vlq::GenParticle ; 
    public:
      GenParticleWithDaughters () {}
      ~GenParticleWithDaughters () {} 
      GenParticleWithDaughters (const GenParticleWithDaughters& p) :
        mom(p.getMom()),
        daus(p.getDaughters())
      {}
      void setMom ( const vlq::GenParticle& p ) { 
        mom.setCharge(p.getCharge()) ;
        mom.setPdgID(p.getPdgID()) ;
        mom.setMomPdgID(p.getMomPdgID());
        mom.setStatus(p.getStatus()) ; 
        mom.setP4(p.getP4()) ; 
      }
      void setDaughters ( const vlq::GenParticleCollection& ps ) { 
        GenParticleCollection::const_iterator it ;
        for ( it = ps.begin(); it != ps.end(); ++it ) {
          GenParticle dau(*it) ; 
          daus.push_back(dau) ; 
        }
      } 
      vlq::GenParticle getMom () const { return  mom; }
      vlq::GenParticleCollection getDaughters () const { return daus ; }
    private: 
      vlq::GenParticle mom ; 
      vlq::GenParticleCollection daus ; 
  }; 

}

#endif 


