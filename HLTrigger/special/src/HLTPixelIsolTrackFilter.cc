#include "HLTrigger/special/interface/HLTPixelIsolTrackFilter.h"

#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidate.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/Common/interface/RefToBase.h"

#include "DataFormats/HLTReco/interface/HLTFilterObject.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"


HLTPixelIsolTrackFilter::HLTPixelIsolTrackFilter(const edm::ParameterSet& iConfig)
{
  candTag_ = iConfig.getUntrackedParameter<edm::InputTag> ("candTag");
  minpttrack = iConfig.getParameter<double> ("MinPtTrack");
  maxptnearby  = iConfig.getParameter<double> ("MaxPtNearby");

  //register your products
  produces<reco::HLTFilterObjectWithRefs>();
}

HLTPixelIsolTrackFilter::~HLTPixelIsolTrackFilter(){}

bool HLTPixelIsolTrackFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  // The Filter object
  std::auto_ptr<reco::HLTFilterObjectWithRefs> filterproduct (new reco::HLTFilterObjectWithRefs(path(),module()));

  // Ref to Candidate object to be recorded in filter object
  edm::RefToBase<reco::Candidate> candref;

  // get hold of filtered candidates
  edm::Handle<reco::IsolatedPixelTrackCandidateCollection> recotrackcands;
  iEvent.getByLabel(candTag_,recotrackcands);

  reco::IsolatedPixelTrackCandidateCollection::const_iterator cands_it;
  reco::IsolatedPixelTrackCandidateCollection::const_iterator cands_beg=recotrackcands->begin();
  reco::IsolatedPixelTrackCandidateCollection::const_iterator cands_end=recotrackcands->end();

  //Filtering

  unsigned int n=0;

  for (cands_it=cands_beg; cands_it<cands_end; cands_it++)
    {
      candref=edm::RefToBase<reco::Candidate>(reco::IsolatedPixelTrackCandidateRef(recotrackcands,distance(cands_beg,cands_it)));

      if ((cands_it->maxPtPxl()<maxptnearby)&&
	  (cands_it->pt()>minpttrack))
	{
	  filterproduct->putParticle(candref);
	  n++;
	}
    }
  
  
  bool accept(n>0);

  iEvent.put(filterproduct);

  return accept;
}
	  
