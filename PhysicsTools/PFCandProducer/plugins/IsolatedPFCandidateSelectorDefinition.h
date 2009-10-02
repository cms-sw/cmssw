#ifndef PhysicsTools_PFCandProducer_IsolatedPFCandidateSelectorDefinition
#define PhysicsTools_PFCandProducer_IsolatedPFCandidateSelectorDefinition

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "PhysicsTools/PFCandProducer/plugins/PFCandidateSelectorDefinition.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "FWCore/Utilities/interface/Exception.h"

struct IsolatedPFCandidateSelectorDefinition : public PFCandidateSelectorDefinition {

  typedef edm::ValueMap<double> IsoMap;

  IsolatedPFCandidateSelectorDefinition ( const edm::ParameterSet & cfg ) :
    isolationValueMapLabels_(cfg.getParameter< std::vector<edm::InputTag> >("isolationValueMaps") ),
    isolationCuts_(cfg.getParameter< std::vector<double> >("isolationCuts")) { 

    if( isolationCuts_.size() != isolationValueMapLabels_.size() )
      throw cms::Exception("BadConfiguration")<<"the vector of isolation ValueMaps and the vector of the corresponding cuts must have the same size."<<std::endl;
  }

  void select( const HandleToCollection & hc, 
	       const edm::Event & e,
	       const edm::EventSetup& s) {
    selected_.clear();
    
/*     assert( hc.isValid() ); */

    // read all isolation value maps
    std::vector< edm::Handle<IsoMap> > 
      isoMaps(isolationValueMapLabels_.size());
    for(unsigned iMap = 0; iMap<isolationValueMapLabels_.size(); ++iMap) {
      e.getByLabel(isolationValueMapLabels_[iMap], isoMaps[iMap]);
    }

    
    unsigned key=0;
    //    for( unsigned i=0; i<collection->size(); i++ ) {
    for( collection::const_iterator pfc = hc->begin(); 
         pfc != hc->end(); ++pfc, ++key) {
      reco::PFCandidateRef candidate(hc,key);

      bool passed = true;
      for(unsigned iMap = 0; iMap<isoMaps.size(); ++iMap) {
	
	const IsoMap & isoMap = *(isoMaps[iMap]);
	
	double val = isoMap[candidate];
	double cut = isolationCuts_[iMap];

	if (val>cut) {
	  passed = false;
	  break; 
	  
	}
      }

      if(passed) {
	// passed all cuts, selected
	selected_.push_back( reco::PFCandidate(*pfc) );
	reco::PFCandidatePtr ptrToMother( hc, key );
	selected_.back().setSourceCandidatePtr( ptrToMother );
      }
    }
  }

private:
  std::vector<edm::InputTag> isolationValueMapLabels_;
  std::vector<double> isolationCuts_;
};

#endif
