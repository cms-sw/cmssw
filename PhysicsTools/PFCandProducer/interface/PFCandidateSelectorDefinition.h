#ifndef RecoParticleFlow_PFPAT_PFCandidateSelector
#define RecoParticleFlow_PFPAT_PFCandidateSelector

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

struct PFCandidateSelector {

  typedef reco::PFCandidateCollection collection;
  typedef std::vector<const reco::PFCandidate *> container;
  typedef container::const_iterator const_iterator;

  PFCandidateSelector ( const edm::ParameterSet & cfg ) :
    ptMin_( cfg.getParameter<double>( "ptMin" ) ) { }

  const_iterator begin() const { return selected_.begin(); }

  const_iterator end() const { return selected_.end(); }

  void select( const collection & c, const edm::Event & ) {
    selected_.clear();
    for( reco::PFCandidateCollection::const_iterator trk = c.begin(); 
         trk != c.end(); ++ trk )
      if ( trk->pt() > ptMin_ ) selected_.push_back( & * trk );
  }

  size_t size() const { return selected_.size(); }

private:
  container selected_;
  double ptMin_;
};

#endif
