#ifndef CommonTools_ParticleFlow_IsolatedPFCandidateSelectorDefinition
#define CommonTools_ParticleFlow_IsolatedPFCandidateSelectorDefinition

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "CommonTools/ParticleFlow/interface/PFCandidateSelectorDefinition.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace pf2pat {

  class IsolatedPFCandidateSelectorDefinition : public PFCandidateSelectorDefinition {

  public:
    typedef edm::ValueMap<double> IsoMap;

    IsolatedPFCandidateSelectorDefinition ( const edm::ParameterSet & cfg ) :
      isolationValueMapLabels_(cfg.getParameter< std::vector<edm::InputTag> >("isolationValueMaps") ),
      isRelative_(cfg.getParameter<bool>("isRelative")),
      isCombined_(cfg.getParameter<bool>("isCombined")),
      isolationCuts_(cfg.getParameter< std::vector<double> >("isolationCuts")),
      combinedIsolationCut_(cfg.getParameter<double>("combinedIsolationCut")) { 
    

      if( isolationCuts_.size() != isolationValueMapLabels_.size() )
	throw cms::Exception("BadConfiguration")<<"the vector of isolation ValueMaps and the vector of the corresponding cuts must have the same size."<<std::endl;
    }

    void select( const HandleToCollection & hc, 
		 const edm::EventBase & e,
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
	double isoSum=0.0;
	for(unsigned iMap = 0; iMap<isoMaps.size(); ++iMap) {
	
	  const IsoMap & isoMap = *(isoMaps[iMap]);
	
	  double val = isoMap[candidate];
	  double cut = isolationCuts_[iMap];
	  if(isRelative_ && candidate->pt()>0.0) val/=candidate->pt();
	  isoSum+=val;
	  //std::cout << "val " << iMap << " = " << val << std::endl;

	  if ( !isCombined_ && val>cut ) {
	    passed = false;
	    break; 	  
	  }
	}

	if ( isCombined_ && isoSum>combinedIsolationCut_ )
	  {
	    passed = false;
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
    bool isRelative_, isCombined_;
    std::vector<double> isolationCuts_;
    double combinedIsolationCut_;
  };

}

#endif
