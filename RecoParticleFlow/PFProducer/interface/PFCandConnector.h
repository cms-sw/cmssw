#ifndef PFProducer_PFCandConnector_H_
#define PFProducer_PFCandConnector_H_

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

// \author : V. Roberfroid
// \date : February 2008

class PFCandConnector {
    
    public :
       
       PFCandConnector( ) { pfC_ = std::auto_ptr<reco::PFCandidateCollection>(new reco::PFCandidateCollection); }
       
       std::auto_ptr<reco::PFCandidateCollection> connect(std::auto_ptr<reco::PFCandidateCollection>& pfCand);
 
    private :
       bool shouldBeLinked( const reco::PFCandidate& pf1, const reco::PFCandidate& pf2) const;

       void link( reco::PFCandidate& pf1, reco::PFCandidate& pf2) const;

       bool isSecondary( const reco::PFCandidate& pf ) const;

       // collection of primary PFCandidate containing secondary
       std::auto_ptr<reco::PFCandidateCollection> pfC_;
};

#endif
