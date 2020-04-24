//
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
      ~PFParticle() override {}

      /// required reimplementation of the Candidate's clone method
      PFParticle * clone() const override { return new PFParticle(*this); }
    
  };
  

}

#endif
