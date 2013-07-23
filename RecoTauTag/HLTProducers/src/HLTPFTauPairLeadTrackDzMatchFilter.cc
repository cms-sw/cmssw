// $Id: HLTPFTauPairLeadTrackDzMatchFilter.cc,v 1.3 2012/02/14 06:03:58 gruen Exp $

#include "HLTPFTauPairLeadTrackDzMatchFilter.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "DataFormats/Math/interface/deltaR.h"


HLTPFTauPairLeadTrackDzMatchFilter::HLTPFTauPairLeadTrackDzMatchFilter(const edm::ParameterSet& conf) : HLTFilter(conf){
 
  tauSrc_	        = conf.getParameter<edm::InputTag>("tauSrc");
  tauMinPt_	        = conf.getParameter<double>("tauMinPt");
  tauMaxEta_	        = conf.getParameter<double>("tauMaxEta");
  tauMinDR_	        = conf.getParameter<double>("tauMinDR");
  tauLeadTrackMaxDZ_	= conf.getParameter<double>("tauLeadTrackMaxDZ");
  triggerType_          = conf.getParameter<int>("triggerType");

  // set the minimum DR between taus, so that one never has a chance 
  // to create a pair out of the same Tau replicated with two
  // different vertices
  if (tauMinDR_ < 0.1) tauMinDR_ = 0.1;
  
}

HLTPFTauPairLeadTrackDzMatchFilter::~HLTPFTauPairLeadTrackDzMatchFilter(){}

void HLTPFTauPairLeadTrackDzMatchFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions){

  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);

  desc.add<edm::InputTag>("tauSrc",edm::InputTag("hltPFTaus") );
  desc.add<double>("tauMinPt",5.0);
  desc.add<double>("tauMaxEta",2.5);
  desc.add<double>("tauMinDR",0.1);
  desc.add<double>("tauLeadTrackMaxDZ",0.2);
  desc.add<int>("triggerType",trigger::TriggerTau);
  descriptions.add("hltPFTauPairLeadTrackDzMatchFilter",desc);
}

bool HLTPFTauPairLeadTrackDzMatchFilter::hltFilter(edm::Event& ev, const edm::EventSetup& es, trigger::TriggerFilterObjectWithRefs& filterproduct){

  using namespace std;
  using namespace reco;
  
  // The resuilting filter object to store in the Event
  if(saveTags() ) filterproduct.addCollectionTag(tauSrc_);

  // Ref to Candidate object to be recorded in the filter object
  PFTauRef ref;

  // Pick up taus
  edm::Handle<PFTauCollection> tausHandle;
  ev.getByLabel( tauSrc_, tausHandle );
  const PFTauCollection & taus = *tausHandle;
  const size_t n_taus = taus.size();

  // Combine taus into pairs and check the dz matching
  size_t npairs = 0, nfail_dz = 0;
  if(n_taus > 1) for(size_t t1 = 0; t1 < n_taus; ++t1) { 
    if( taus[t1].leadPFChargedHadrCand().isNull() ||
	taus[t1].leadPFChargedHadrCand()->trackRef().isNull() ||
        taus[t1].pt() < tauMinPt_ || 
	std::abs(taus[t1].eta() ) > tauMaxEta_ ) 
      continue;
    
    float mindz = 99.f;
    for(size_t t2 = t1+1; t2 < n_taus; ++t2){
      if( taus[t2].leadPFChargedHadrCand().isNull() ||
	  taus[t2].leadPFChargedHadrCand()->trackRef().isNull() ||
	  taus[t2].pt() < tauMinPt_ || 
	  std::abs(taus[t2].eta() ) > tauMaxEta_ ) 
	continue;

      float dr2 = reco::deltaR2(taus[t1].eta(), taus[t1].phi(),
				taus[t2].eta(), taus[t2].phi() );
      float dz = ( taus[t1].leadPFChargedHadrCand()->trackRef()->vz() -
		   taus[t2].leadPFChargedHadrCand()->trackRef()->vz() );

      // skip pairs of taus that are close
      if ( dr2 < tauMinDR_*tauMinDR_ ) {
	continue;
      }
      
      if (std::abs(dz) < std::abs(mindz)) mindz = dz;

      // do not form a pair if dz is too large
      if ( std::abs(dz) > tauLeadTrackMaxDZ_ ) {
	++nfail_dz;
	continue;
      }
      
      // add references to both jets
      ref = PFTauRef(tausHandle, t1);
      filterproduct.addObject(triggerType_, ref);
      
      ref = PFTauRef(tausHandle, t2);
      filterproduct.addObject(triggerType_, ref);

      ++npairs;
      
    }

  }

  // return truth if at least one good pair found
  return npairs>0;

}
