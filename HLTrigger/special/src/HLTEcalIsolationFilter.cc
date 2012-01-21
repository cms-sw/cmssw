#include "HLTrigger/special/interface/HLTEcalIsolationFilter.h"

#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidate.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/Common/interface/RefToBase.h"

//#include "DataFormats/HLTReco/interface/HLTFilterObject.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"


HLTEcalIsolationFilter::HLTEcalIsolationFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig) 
{
  candTag_ = iConfig.getParameter<edm::InputTag> ("EcalIsolatedParticleSource");
  maxhitout = iConfig.getParameter<int> ("MaxNhitOuterCone");
  maxhitin  = iConfig.getParameter<int> ("MaxNhitInnerCone");
  maxenin = iConfig.getParameter<double> ("MaxEnergyInnerCone");
  maxenout = iConfig.getParameter<double> ("MaxEnergyOuterCone");
  maxetacand = iConfig.getParameter<double> ("MaxEtaCandidate");  
}

HLTEcalIsolationFilter::~HLTEcalIsolationFilter(){}

bool HLTEcalIsolationFilter::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct)
{

  // Ref to Candidate object to be recorded in filter object
  edm::Ref<reco::IsolatedPixelTrackCandidateCollection> candref;

  // get hold of filtered candidates
  edm::Handle<reco::IsolatedPixelTrackCandidateCollection> ecIsolCands;
  iEvent.getByLabel(candTag_,ecIsolCands);

  //Filtering

  unsigned int n=0;
  for (unsigned int i=0; i<ecIsolCands->size(); i++)
    {
      candref = edm::Ref<reco::IsolatedPixelTrackCandidateCollection>(ecIsolCands, i);
	
      if ((candref->nHitIn()<=maxhitin)&&(candref->nHitOut()<=maxhitout)&&(candref->energyOut()<maxenout)&&(candref->energyIn()<maxenin)&&fabs(candref->eta())<maxetacand)
	{
	  filterproduct.addObject(trigger::TriggerTrack, candref);
	  n++;
	}
    }
  
  
  bool accept(n>0);

  return accept;

}
	  
