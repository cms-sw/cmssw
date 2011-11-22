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
  hltGTseedlabel_      = iConfig.getParameter<edm::InputTag> ("L1GTSeedLabel");
  minDeltaPtL1Jet_     = iConfig.getParameter<double> ("MinDeltaPtL1Jet");
  minpttrack_          = iConfig.getParameter<double> ("MinPtTrack");
  maxptnearby_         = iConfig.getParameter<double> ("MaxPtNearby");
  maxetatrack_         = iConfig.getParameter<double> ("MaxEtaTrack");
  minetatrack_         = iConfig.getParameter<double> ("MinEtaTrack");
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

  //find leading L1 jet:
  double ptTriggered  = -10;

  edm::Handle<trigger::TriggerFilterObjectWithRefs> l1trigobj;
  iEvent.getByLabel(hltGTseedlabel_, l1trigobj);
  
  std::vector< edm::Ref<l1extra::L1JetParticleCollection> > l1tauobjref;
  std::vector< edm::Ref<l1extra::L1JetParticleCollection> > l1jetobjref;
  std::vector< edm::Ref<l1extra::L1JetParticleCollection> > l1forjetobjref;
  
  l1trigobj->getObjects(trigger::TriggerL1TauJet, l1tauobjref);
  l1trigobj->getObjects(trigger::TriggerL1CenJet, l1jetobjref);
  l1trigobj->getObjects(trigger::TriggerL1ForJet, l1forjetobjref);
  
  for (unsigned int p=0; p<l1tauobjref.size(); p++)
      if (l1tauobjref[p]->pt() > ptTriggered)
	  ptTriggered = l1tauobjref[p]->pt(); 
  for (unsigned int p=0; p<l1jetobjref.size(); p++)
      if (l1jetobjref[p]->pt() > ptTriggered)
	  ptTriggered = l1jetobjref[p]->pt();
  for (unsigned int p=0; p<l1forjetobjref.size(); p++)
      if (l1forjetobjref[p]->pt() > ptTriggered)
          ptTriggered = l1forjetobjref[p]->pt();

  int n=0;
  for (unsigned int i=0; i<recotrackcands->size(); i++)
    {
      candref = edm::Ref<reco::IsolatedPixelTrackCandidateCollection>(recotrackcands, i);
      
      // cut on deltaPT
      if (ptTriggered-candref->pt()<minDeltaPtL1Jet_) continue;
      
      // select on transverse momentum
      if (!filterE_&&(candref->maxPtPxl()<maxptnearby_)&&
	  (candref->pt()>minpttrack_)&&fabs(candref->track()->eta())<maxetatrack_&&fabs(candref->track()->eta())>minetatrack_)
	{
	  filterproduct->addObject(trigger::TriggerTrack, candref);
	  n++;
	}

      // select on momentum
      if (filterE_){
	if ((candref->maxPtPxl()<maxptnearby_)&&((candref->pt())*cosh(candref->track()->eta())>minEnergy_)&&fabs(candref->track()->eta())<maxetatrack_&&fabs(candref->track()->eta())>minetatrack_)
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
	  
