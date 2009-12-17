#include "HLTrigger/special/interface/HLTPixelIsolTrackFilter.h"

#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidate.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/Common/interface/RefToBase.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"

//#include "DataFormats/HLTReco/interface/HLTFilterObject.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"


HLTPixelIsolTrackFilter::HLTPixelIsolTrackFilter(const edm::ParameterSet& iConfig)
{
  candTag_             = iConfig.getParameter<edm::InputTag> ("candTag");
  minpttrack           = iConfig.getParameter<double> ("MinPtTrack");
  maxptnearby          = iConfig.getParameter<double> ("MaxPtNearby");
  maxetatrack          = iConfig.getParameter<double> ("MaxEtaTrack");
  minetatrack          = iConfig.getParameter<double> ("MinEtaTrack");
  filterE_             = iConfig.getParameter<bool> ("filterTrackEnergy");
  minEnergy_           = iConfig.getParameter<double>("MinEnergyTrack");
  nMaxTrackCandidates_ = iConfig.getParameter<int>("NMaxTrackCandidates");
  dropMultiL2Event_    = iConfig.getParameter<bool> ("DropMultiL2Event");
  //register your products
  produces<trigger::TriggerFilterObjectWithRefs>();
}

HLTPixelIsolTrackFilter::~HLTPixelIsolTrackFilter(){}

bool HLTPixelIsolTrackFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  // The Filter object
  std::auto_ptr<trigger::TriggerFilterObjectWithRefs> filterproduct (new trigger::TriggerFilterObjectWithRefs(path(),module()));

  // Ref to Candidate object to be recorded in filter object
  edm::Ref<reco::IsolatedPixelTrackCandidateCollection> candref;

  // get hold of filtered candidates
  edm::Handle<reco::IsolatedPixelTrackCandidateCollection> recotrackcands;
  iEvent.getByLabel(candTag_,recotrackcands);

  //Filtering

  int n=0;
  for (unsigned int i=0; i<recotrackcands->size(); i++)
    {
      candref = edm::Ref<reco::IsolatedPixelTrackCandidateCollection>(recotrackcands, i);

      // select on transverse momentum
      if (!filterE_&&(candref->maxPtPxl()<maxptnearby)&&
	  (candref->pt()>minpttrack)&&fabs(candref->track()->eta())<maxetatrack&&fabs(candref->track()->eta())>minetatrack)
	{
	  filterproduct->addObject(trigger::TriggerTrack, candref);
	  n++;
	}

      // select on momentum
      if (filterE_){
	if ((candref->maxPtPxl()<maxptnearby)&&((candref->pt())*cosh(candref->track()->eta())>minEnergy_)&&fabs(candref->track()->eta())<maxetatrack&&fabs(candref->track()->eta())>minetatrack)
	  {
	    filterproduct->addObject(trigger::TriggerTrack, candref);
	    n++;
	  }
      }

      // stop looping over tracks if max number is reached
      if(!dropMultiL2Event_ && n>=nMaxTrackCandidates_) break; 

    } // loop over tracks
  
  
  bool accept(n>0);

  if( dropMultiL2Event_ && n>nMaxTrackCandidates_ ) accept=false;

  filterproduct->addCollectionTag(candTag_);

  iEvent.put(filterproduct);

  return accept;

}
	  
