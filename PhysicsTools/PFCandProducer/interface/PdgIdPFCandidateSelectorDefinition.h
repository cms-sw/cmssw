#ifndef RecoParticleFlow_PFPAT_PdgIdPFCandidateSelectorDefinition
#define RecoParticleFlow_PFPAT_PdgIdPFCandidateSelectorDefinition

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

struct PdgIdPFCandidateSelectorDefinition {

  typedef reco::PFCandidateCollection collection;
  typedef edm::Handle< collection > HandleToCollection;
  typedef std::vector< reco::PFCandidate *> container;
  typedef container::const_iterator const_iterator;

  PdgIdPFCandidateSelectorDefinition ( const edm::ParameterSet & cfg ) :
  pdgIds_( cfg.getParameter< std::vector<int> >( "pdgId" ) ) { }

  const_iterator begin() const { return selected_.begin(); }

  const_iterator end() const { return selected_.end(); }

  void select( const HandleToCollection & hc, 
	       const edm::Event & e,
	       const edm::EventSetup& s) {
    selected_.clear();
    
    assert( hc.isValid() );
    
    collection col = *hc;
    unsigned key=0;
    for( collection::const_iterator trk = col.begin(); 
         trk != col.end(); ++trk, ++key) {
      
      for(unsigned iId=0; iId<pdgIds_.size(); iId++) {
	if ( trk->pdgId() == pdgIds_[iId] ) {
	  selected_.push_back( new reco::PFCandidate(*trk) );
	  reco::PFCandidateRef refToMother( hc, key );
	  selected_.back()->setSourceRef( refToMother );
	  break;
	}
      }
    }
  }

  size_t size() const { return selected_.size(); }

private:
  container selected_;
  std::vector<int> pdgIds_;
};

#endif
