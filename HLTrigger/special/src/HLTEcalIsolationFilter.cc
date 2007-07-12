#include "HLTrigger/special/interface/HLTEcalIsolationFilter.h"

#include "DataFormats/HcalIsolatedTrack/interface/EcalIsolatedParticleCandidate.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/Common/interface/RefToBase.h"

#include "DataFormats/HLTReco/interface/HLTFilterObject.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"


HLTEcalIsolationFilter::HLTEcalIsolationFilter(const edm::ParameterSet& iConfig)
{
  candTag_ = iConfig.getUntrackedParameter<edm::InputTag> ("EcalIsolatedParticleSource");
  minen = iConfig.getParameter<double> ("MinEnergy");
  maxennearby  = iConfig.getParameter<double> ("MaxEnergyNearby");

  //register your products
  produces<reco::HLTFilterObjectWithRefs>();
}

HLTEcalIsolationFilter::~HLTEcalIsolationFilter(){}

bool HLTEcalIsolationFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  // The Filter object
  std::auto_ptr<reco::HLTFilterObjectWithRefs> filterproduct (new reco::HLTFilterObjectWithRefs(path(),module()));

  // Ref to Candidate object to be recorded in filter object
  edm::RefToBase<reco::Candidate> candref;

  // get hold of filtered candidates
  edm::Handle<reco::EcalIsolatedParticleCandidateCollection> ecIsolCands;
  iEvent.getByLabel(candTag_,ecIsolCands);

  reco::EcalIsolatedParticleCandidateCollection::const_iterator cands_it;
  reco::EcalIsolatedParticleCandidateCollection::const_iterator cands_beg=ecIsolCands->begin();
  reco::EcalIsolatedParticleCandidateCollection::const_iterator cands_end=ecIsolCands->end();

  //Filtering

  unsigned int n=0;

  for (cands_it=cands_beg; cands_it<cands_end; cands_it++)
    {
      candref=edm::RefToBase<reco::Candidate>(reco::EcalIsolatedParticleCandidateRef(ecIsolCands,distance(cands_beg,cands_it)));

      if ((cands_it->enMaxNear()<maxennearby)&&
	  (cands_it->energy()>minen))
	{
	  filterproduct->putParticle(candref);
	  n++;
	}
    }
  
  
  bool accept(n>0);

  iEvent.put(filterproduct);

  return accept;
}
	  
