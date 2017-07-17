#ifndef CommonTools_ParticleFlow_IsolatedPFCandidateSelectorDefinition
#define CommonTools_ParticleFlow_IsolatedPFCandidateSelectorDefinition

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "CommonTools/ParticleFlow/interface/PFCandidateSelectorDefinition.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/transform.h"

namespace pf2pat {

  class IsolatedPFCandidateSelectorDefinition : public PFCandidateSelectorDefinition {

  public:
    typedef edm::ValueMap<double> IsoMap;

    IsolatedPFCandidateSelectorDefinition ( const edm::ParameterSet & cfg, edm::ConsumesCollector && iC ) :
      isolationValueMapChargedTokens_(edm::vector_transform(cfg.getParameter< std::vector<edm::InputTag> >("isolationValueMapsCharged"), [&](edm::InputTag const & tag){return iC.consumes<IsoMap>(tag);}) ),
      isolationValueMapNeutralTokens_(edm::vector_transform(cfg.getParameter< std::vector<edm::InputTag> >("isolationValueMapsNeutral"), [&](edm::InputTag const & tag){return iC.consumes<IsoMap>(tag);}) ),
      doDeltaBetaCorrection_(cfg.getParameter<bool>("doDeltaBetaCorrection")),
      deltaBetaIsolationValueMapToken_(iC.mayConsume<IsoMap>(cfg.getParameter< edm::InputTag >("deltaBetaIsolationValueMap") ) ),
      deltaBetaFactor_(cfg.getParameter<double>("deltaBetaFactor")),
      isRelative_(cfg.getParameter<bool>("isRelative")),
      isolationCut_(cfg.getParameter<double>("isolationCut")) {}



    void select( const HandleToCollection & hc,
		 const edm::Event & e,
		 const edm::EventSetup& s) {
      selected_.clear();


      // read all charged isolation value maps
      std::vector< edm::Handle<IsoMap> >
	isoMapsCharged(isolationValueMapChargedTokens_.size());
      for(unsigned iMap = 0; iMap<isolationValueMapChargedTokens_.size(); ++iMap) {
	e.getByToken(isolationValueMapChargedTokens_[iMap], isoMapsCharged[iMap]);
      }


      // read all neutral isolation value maps
      std::vector< edm::Handle<IsoMap> >
	isoMapsNeutral(isolationValueMapNeutralTokens_.size());
      for(unsigned iMap = 0; iMap<isolationValueMapNeutralTokens_.size(); ++iMap) {
	e.getByToken(isolationValueMapNeutralTokens_[iMap], isoMapsNeutral[iMap]);
      }

      edm::Handle<IsoMap> dBetaH;
      if(doDeltaBetaCorrection_) {
	e.getByToken(deltaBetaIsolationValueMapToken_, dBetaH);
      }

      unsigned key=0;
      for( collection::const_iterator pfc = hc->begin();
	   pfc != hc->end(); ++pfc, ++key) {
	reco::PFCandidateRef candidate(hc,key);

	bool passed = true;
	double isoSumCharged=0.0;
	double isoSumNeutral=0.0;

	for(unsigned iMap = 0; iMap<isoMapsCharged.size(); ++iMap) {
	  const IsoMap & isoMap = *(isoMapsCharged[iMap]);
	  double val = isoMap[candidate];
	  isoSumCharged+=val;
	}

	for(unsigned iMap = 0; iMap<isoMapsNeutral.size(); ++iMap) {
	  const IsoMap & isoMap = *(isoMapsNeutral[iMap]);
	  double val = isoMap[candidate];
	  isoSumNeutral+=val;
	}


	if ( doDeltaBetaCorrection_ ) {
	  const IsoMap& isoMap = *dBetaH;
	  double dBetaVal = isoMap[candidate];
	  double dBetaCorIsoSumNeutral = isoSumNeutral + deltaBetaFactor_*dBetaVal;
	  isoSumNeutral = dBetaCorIsoSumNeutral>0 ? dBetaCorIsoSumNeutral : 0; //follow muon POG definition in 2012
	}

	double isoSum=isoSumCharged+isoSumNeutral;

	if( isRelative_ ) {
	  isoSum /= candidate->pt();
	}

	if ( isoSum>isolationCut_ ) {
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
    std::vector<edm::EDGetTokenT<IsoMap> > isolationValueMapChargedTokens_;
    std::vector<edm::EDGetTokenT<IsoMap> > isolationValueMapNeutralTokens_;
    bool                       doDeltaBetaCorrection_;
    edm::EDGetTokenT<IsoMap>              deltaBetaIsolationValueMapToken_;
    double                     deltaBetaFactor_;
    bool                       isRelative_;
    double                     isolationCut_;
  };

}

#endif
