//
// $Id: PFParticle.h,v 1.3 2008/11/28 19:02:15 lowette Exp $
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
  \version  $Id: PFParticle.h,v 1.3 2008/11/28 19:02:15 lowette Exp $
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


// Class definition
namespace pat {


  class PFParticle : public PATObject<reco::PFCandidate> {
    
    public:
    
      /// default constructor
      PFParticle() {}
      /// constructor from ref
      PFParticle(const edm::RefToBase<reco::PFCandidate> & aPFParticle);
      /// destructor
      virtual ~PFParticle() {}

      /// required reimplementation of the Candidate's clone method
      virtual PFParticle * clone() const { return new PFParticle(*this); }
    
  };
  

}

#endif
