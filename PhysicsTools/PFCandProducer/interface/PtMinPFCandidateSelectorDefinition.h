#ifndef RecoParticleFlow_PFPAT_PtMinPFCandidateSelectorDefinition
#define RecoParticleFlow_PFPAT_PtMinPFCandidateSelectorDefinition

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

struct PtMinPFCandidateSelectorDefinition {

  typedef reco::PFCandidateCollection collection;
  typedef edm::Handle< collection > HandleToCollection;
  typedef std::vector< reco::PFCandidate *> container;
  typedef container::const_iterator const_iterator;

  PtMinPFCandidateSelectorDefinition ( const edm::ParameterSet & cfg ) :
  ptMin_( cfg.getParameter< double >( "ptMin" ) ) { }

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

      if( trk->pt() > ptMin_ ) {
	selected_.push_back( new reco::PFCandidate(*trk) );
	reco::PFCandidateRef refToMother( hc, key );
	selected_.back()->setMotherRef( refToMother );
      }
    }
  }

  size_t size() const { return selected_.size(); }

private:
  container selected_;
  double ptMin_;
};

#endif
