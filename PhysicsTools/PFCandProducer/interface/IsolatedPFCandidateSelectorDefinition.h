#ifndef PhysicsTools_PFCandProducer_IsolatedPFCandidateSelectorDefinition
#define PhysicsTools_PFCandProducer_IsolatedPFCandidateSelectorDefinition

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/Common/interface/ValueMap.h"
struct IsolatedPFCandidateSelectorDefinition {

  typedef reco::PFCandidateCollection collection;
  typedef edm::Handle< collection > HandleToCollection;
  typedef std::vector< reco::PFCandidate *> container;
  typedef container::const_iterator const_iterator;
  typedef edm::ValueMap<double> isoFromDepositsMap;

  IsolatedPFCandidateSelectorDefinition ( const edm::ParameterSet & cfg ) :
    isoDepositMap_(cfg.getParameter<edm::InputTag>("IsoDeposit") ),
    isolCut_(cfg.getParameter<double>("IsolationCut")) { }

  const_iterator begin() const { return selected_.begin(); }

  const_iterator end() const { return selected_.end(); }

  void select( const HandleToCollection & hc, 
	       const edm::Event & e,
	       const edm::EventSetup& s) {
    selected_.clear();
    
    assert( hc.isValid() );

    edm::Handle<isoFromDepositsMap  > iso;
    e.getByLabel(isoDepositMap_,iso);
    const isoFromDepositsMap & qq= *(iso);
    
    unsigned key=0;
    //    for( unsigned i=0; i<collection->size(); i++ ) {
    for( collection::const_iterator pfc = hc->begin(); 
         pfc != hc->end(); ++pfc, ++key) {
      reco::PFCandidateRef c(hc,key);
      float val = qq[c];
  
    
      if (val<isolCut_) {

	selected_.push_back( new reco::PFCandidate(*pfc) );
	reco::PFCandidatePtr ptrToMother( hc, key );

	selected_.back()->setSourcePtr( ptrToMother );
      }
    }
  }

  size_t size() const { return selected_.size(); }

private:
  edm::InputTag isoDepositMap_;
  double isolCut_;
  container selected_;

};

#endif
