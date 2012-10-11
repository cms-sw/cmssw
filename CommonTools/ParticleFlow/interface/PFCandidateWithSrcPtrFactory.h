#ifndef CommonTools_ParticleFlow_PFCandidateWithSrcPtrFactor_h
#define CommonTools_ParticleFlow_PFCandidateWithSrcPtrFactor_h

/**
  \class    reco::PFCandidateWithSrcPtrFactory PFCandidateWithSrcPtrFactory.h  "CommonTools/ParticleFlow/interface/PFCandidateWithSrcPtrFactory.h"
  \brief    Creates a PFCandidate from an input FwdPtr, and sets the "source" Ptr to the FwdPtr.backPtr 


  \author   Salvatore Rappoccio
*/

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

namespace reco {
  class PFCandidateWithSrcPtrFactory : public std::unary_function<reco::PFCandidate, edm::FwdPtr<reco::PFCandidate> > {
  public :
    reco::PFCandidate operator()( edm::FwdPtr<reco::PFCandidate> const & input ) const {
      reco::PFCandidate output( *input );
      for ( unsigned int isource = 0; isource < input->numberOfSourceCandidatePtrs(); ++isource ) {
	edm::Ptr<reco::PFCandidate> ptr (input->sourceCandidatePtr(isource) );
	output.setSourceCandidatePtr( ptr );
      }
      output.setSourceCandidatePtr( input.backPtr() );
      return output; 
    }
  };
}

#endif
