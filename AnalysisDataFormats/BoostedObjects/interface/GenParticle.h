#ifndef ANALYSISDATAFORMATS_BOOSTEDOBJECTS_GENPARTICLE_HH
#define ANALYSISDATAFORMATS_BOOSTEDOBJECTS_GENPARTICLE_HH

#include "AnalysisDataFormats/BoostedObjects/interface/Candidate.h"

namespace vlq {
  class GenParticle ;
  typedef std::vector<GenParticle> GenParticleCollection ; 
  class GenParticle : public Candidate {
    public: 
      GenParticle () : charge_(-999999), pdgID_(-999999), momPdgID_(-999999), status_(-999999) {} 
      ~GenParticle () {} 
      GenParticle (const vlq::GenParticle& p ) :
        charge_(p.getCharge()),
        pdgID_(p.getPdgID()),
        momPdgID_(p.getMomPdgID()),
        status_(p.getStatus())  
      { 
        this->setP4( p.getP4() ) ; 
      } 

      int getCharge () const {return charge_ ; }
      int getPdgID () const { return pdgID_ ; }
      int getMomPdgID () const { return momPdgID_ ; }
      int getStatus () const { return status_ ; } 

      void setCharge ( const int charge ) {charge_ = charge ; }
      void setPdgID ( const int pdgID )  { pdgID_ = pdgID ; }
      void setMomPdgID ( const int momPdgID ) { momPdgID_ = momPdgID ; }
      void setStatus ( const int status ) { status_ = status ; } 

    protected:
      int charge_ ; 
      int pdgID_ ;
      int momPdgID_ ;
      int status_ ; 
  }; 
}
#endif
