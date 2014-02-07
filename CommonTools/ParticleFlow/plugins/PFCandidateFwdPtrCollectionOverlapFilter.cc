#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/FwdPtr.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "CommonTools/ParticleFlow/plugins/PFCandidateFwdPtrCollectionOverlapFilter.h"
#include "CommonTools/ParticleFlow/interface/PFCandidateWithSrcPtrFactory.h"
#include <vector>

PFCandidateFwdPtrCollectionOverlapFilter::PFCandidateFwdPtrCollectionOverlapFilter( edm::ParameterSet const & params ) :
  srcToken_( consumes< std::vector< edm::FwdPtr<reco::PFCandidate> > >( params.getParameter<edm::InputTag>("src") ) ),
  srcViewToken_( mayConsume< edm::View<reco::PFCandidate>  >( params.getParameter<edm::InputTag>("src") ) ),
  overlapToken_( consumes< edm::View<reco::Candidate>  >( params.getParameter<edm::InputTag>("overlapCollection") ) ),
  filter_(params.getParameter<bool>("filter")), 
  makeClones_(params.getParameter<bool>("makeClones")),
  maxDeltaR_(params.getParameter<double>("maxDeltaR")),
  maxDPtRel_(params.getParameter<double>("maxDPtRel"))
{
  produces< std::vector< edm::FwdPtr<reco::PFCandidate> > > ();
  if ( makeClones_ ) {
    produces< std::vector<reco::PFCandidate> > ();
  }
}

void PFCandidateFwdPtrCollectionOverlapFilter::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
   edm::ParameterSetDescription desc;
   desc.add<edm::InputTag>("src", edm::InputTag(""));
   desc.add<edm::InputTag>("overlapCollection", edm::InputTag(""));
   desc.add<bool>("filter", false);
   desc.add<bool>("makeClones", false);
   desc.add<double>("maxDeltaR", 0.01);
   desc.add<double>("maxDPtRel", -1.0);
   descriptions.add("pfCandidateOverlapFilter", desc);
}

bool PFCandidateFwdPtrCollectionOverlapFilter::filter(edm::Event & iEvent, const edm::EventSetup& iSetup) {

  std::auto_ptr< std::vector< edm::FwdPtr<reco::PFCandidate> > > pOutput ( new std::vector<edm::FwdPtr<reco::PFCandidate> > );
  std::auto_ptr< std::vector<reco::PFCandidate> > pClones ( new std::vector<reco::PFCandidate> );

  edm::Handle< std::vector< edm::FwdPtr<reco::PFCandidate> > > hSrcAsFwdPtr;
  edm::Handle< edm::View<reco::PFCandidate> > hSrcAsView;
  bool foundAsFwdPtr = iEvent.getByToken( srcToken_, hSrcAsFwdPtr );
  if ( !foundAsFwdPtr ) {
    iEvent.getByToken( srcViewToken_, hSrcAsView );
  }
  edm::Handle< edm::View<reco::Candidate> > hOverlap;
  iEvent.getByToken( overlapToken_, hOverlap );

  // First try to access as a View<reco::PFCandidate>. 
  // If not a View<reco::PFCandidate>, look as a vector<FwdPtr<reco::PFCandidate> >
  if ( !foundAsFwdPtr ) {
    for ( edm::View<reco::PFCandidate>::const_iterator ibegin = hSrcAsView->begin(),
	    iend = hSrcAsView->end(),
	    i = ibegin; i!= iend; ++i ) {
      // loop over candidates to check for overlap
      bool overlap = false;
      for ( edm::View<reco::Candidate>::const_iterator jbegin = hOverlap->begin(),
	      jend = hOverlap->end(),
	      j = jbegin; j!= jend; ++j ) {
	if ( isOverlap(*i,*j) ) {
	  overlap = true;
	  break;
	}
      }
      if ( !overlap ) {
	pOutput->push_back( edm::FwdPtr<reco::PFCandidate>( hSrcAsView->ptrAt( i - ibegin ), hSrcAsView->ptrAt( i - ibegin ) ) );
	if ( makeClones_ ) {
	  reco::PFCandidateWithSrcPtrFactory factory;
	  reco::PFCandidate outclone = factory( pOutput->back() );
	  pClones->push_back( outclone );
	}
      }
    }
  } else {
    for ( typename std::vector<edm::FwdPtr<reco::PFCandidate> >::const_iterator ibegin = hSrcAsFwdPtr->begin(),
	    iend = hSrcAsFwdPtr->end(),
	    i = ibegin; i!= iend; ++i ) {
      // loop over candidates to check for overlap
      bool overlap = false;
      for ( edm::View<reco::Candidate>::const_iterator jbegin = hOverlap->begin(),
	      jend = hOverlap->end(),
	      j = jbegin; j!= jend; ++j ) {
	if ( isOverlap(**i,*j) ) {
	  overlap = true;
	  break;
	}
      }
      if ( !overlap ) {
	pOutput->push_back( *i );
	if ( makeClones_ ) {
	  reco::PFCandidateWithSrcPtrFactory factory;
	  reco::PFCandidate outclone = factory( pOutput->back() );
	  pClones->push_back( outclone );
	}
      }
    }
  }

  bool pass = pOutput->size() > 0;
  iEvent.put( pOutput );
  if ( makeClones_ )
    iEvent.put( pClones );
  if ( filter_ )
    return pass;
  else
    return true;

}

bool PFCandidateFwdPtrCollectionOverlapFilter::isOverlap(const reco::PFCandidate& obj1, const reco::Candidate& obj2) const {
  if ( (maxDeltaR_ > 0.) && (deltaR2(obj1,obj2) > maxDeltaR_*maxDeltaR_) ) return false;
  if ( (maxDPtRel_ > 0.) && (fabs(obj1.pt()-obj2.pt())/obj2.pt() > maxDPtRel_) ) return false;
  return true;
}
