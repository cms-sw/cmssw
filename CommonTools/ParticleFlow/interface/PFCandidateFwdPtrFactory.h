#ifndef CommonTools_ParticleFlow_PFCandidateFwdPtrFactory_h
#define CommonTools_ParticleFlow_PFCandidateFwdPtrFactory_h

/**
  \class    reco::PFCandidateFwdPtrFactory PFCandidateFwdPtrFactory.h  "CommonTools/ParticleFlow/interface/PFCandidateFwdPtrFactory.h"
  \brief    Creates a FwdPtr<PFCandidate> from an input PFCandidate. If the PFCandidate has a valid sourceCandidatePtr, that is
            used for the FwdPtr's "backPtr". 


  \author   Salvatore Rappoccio
*/

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

namespace reco {
  class PFCandidateFwdPtrFactory : public std::binary_function<edm::FwdPtr<reco::PFCandidate>, edm::View<reco::PFCandidate>, unsigned int > {
  public :
    edm::FwdPtr<reco::PFCandidate> operator() (edm::View<reco::PFCandidate> const & view, unsigned int i)  const  { 
      edm::Ptr<reco::PFCandidate> ptr = view.ptrAt(i);
      edm::Ptr<reco::PFCandidate> backPtr = ptr;
      if ( ptr.isNonnull() && ptr.isAvailable() && ptr->numberOfSourceCandidatePtrs() > 0 ) {
	edm::Ptr<reco::Candidate> basePtr = ptr->sourceCandidatePtr(0);
	if (basePtr.isNonnull() && basePtr.isAvailable())
	  backPtr = edm::Ptr<reco::PFCandidate>( basePtr );//this cast works only for available stuff
      }
      return edm::FwdPtr<reco::PFCandidate>(ptr,backPtr); 
    }

  };
}

#endif
