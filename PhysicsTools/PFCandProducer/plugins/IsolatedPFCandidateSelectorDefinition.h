#ifndef PhysicsTools_PFCandProducer_IsolatedPFCandidateSelectorDefinition
#define PhysicsTools_PFCandProducer_IsolatedPFCandidateSelectorDefinition

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "PhysicsTools/PFCandProducer/plugins/PFCandidateSelectorDefinition.h"
#include "DataFormats/Common/interface/ValueMap.h"
struct IsolatedPFCandidateSelectorDefinition : public PFCandidateSelectorDefinition {

  typedef edm::ValueMap<double> isoFromDepositsMap;

  IsolatedPFCandidateSelectorDefinition ( const edm::ParameterSet & cfg ) :
    isoDepositMap_(cfg.getParameter<edm::InputTag>("IsoDeposit") ),
    isolCut_(cfg.getParameter<double>("IsolationCut")) { }

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

	selected_.push_back( reco::PFCandidate(*pfc) );
	reco::PFCandidatePtr ptrToMother( hc, key );

	selected_.back().setSourceCandidatePtr( ptrToMother );
      }
    }
  }

private:
  edm::InputTag isoDepositMap_;
  double isolCut_;
};

#endif
