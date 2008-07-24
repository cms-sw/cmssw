//
// $Id: PFParticle.h,v 1.4 2008/06/03 22:28:07 gpetrucc Exp $
//

#ifndef DataFormats_PatCandidates_PFParticle_h
#define DataFormats_PatCandidates_PFParticle_h

/**
  \class    pat::PFParticle PFParticle.h "DataFormats/PatCandidates/interface/PFParticle.h"
  \brief    Analysis-level class for reconstructed particles

   PFParticle is the equivalent of reco::PFCandidate in the PAT namespace, 
   to be used in the analysis. All the PFCandidates that are not used as 
   isolated leptons or photons, or inside jets, end up as pat::PFParticles.

  \author   Colin Bernet
  \version  $Id: PFParticle.h,v 1.4 2008/06/03 22:28:07 gpetrucc Exp $
*/

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/PatCandidates/interface/PATObject.h"


namespace pat {


  typedef reco::PFCandidate PFParticleType;
  

  class PFParticle : public PATObject<PFParticleType> {
    
  public:
    
    PFParticle() {}

    PFParticle(const edm::RefToBase<PFParticleType> & aPFParticle);
    
    virtual PFParticle * clone() const { return new PFParticle(*this); }
    
  };
  

}

#endif
