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
  candTag_ = iConfig.getParameter<edm::InputTag> ("candTag");
  minpttrack = iConfig.getParameter<double> ("MinPtTrack");
  maxptnearby  = iConfig.getParameter<double> ("MaxPtNearby");
  maxetatrack  = iConfig.getParameter<double> ("MaxEtaTrack");
  filterE_ = iConfig.getParameter<bool> ("filterTrackEnergy");
  minEnergy_=iConfig.getParameter<double>("MinEnergyTrack");

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

  unsigned int n=0;

  for (unsigned int i=0; i<recotrackcands->size(); i++)
    {
      candref = edm::Ref<reco::IsolatedPixelTrackCandidateCollection>(recotrackcands, i);

      if (!filterE_&&(candref->maxPtPxl()<maxptnearby)&&
	  (candref->pt()>minpttrack)&&fabs(candref->track()->eta())<maxetatrack)
	{
	  filterproduct->addObject(trigger::TriggerTrack, candref);
	  n++;
	}
      if (filterE_){
      if ((candref->maxPtPxl()<maxptnearby)&&((candref->pt())*cosh(candref->track()->eta())>minEnergy_)&&fabs(candref->track()->eta())<maxetatrack)
        {
          filterproduct->addObject(trigger::TriggerTrack, candref);
          n++;
        }
	}

    }
  
  
  bool accept(n>0);

  iEvent.put(filterproduct);

  return accept;

}
	  
