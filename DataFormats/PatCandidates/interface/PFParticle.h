//
// $Id: PFParticle.h,v 1.1 2008/07/24 12:43:52 cbern Exp $
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
  \version  $Id: PFParticle.h,v 1.1 2008/07/24 12:43:52 cbern Exp $
*/

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/PatCandidates/interface/PATObject.h"


// Define typedefs for convenience
namespace pat {
  class PFParticle;
  typedef std::vector<PFParticle>              PFParticleCollection; 
  typedef edm::Ref<PFParticleCollection>       PFParticleRef; 
  typedef edm::RefVector<PFParticleCollection> PFParticleRefVector; 
}

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
