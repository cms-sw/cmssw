/** \class HLTElectronPixelMatchFilter
 *
 * $Id: HLTElectronPixelMatchFilter.cc,v 1.10 2007/12/07 14:41:33 ghezzi Exp $
 *
 *  \author Monica Vazquez Acosta (CERN)
 *
 */

#include "HLTrigger/Egamma/interface/HLTElectronPixelMatchFilter.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Common/interface/AssociationMap.h"


#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/ElectronPixelSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronPixelSeedFwd.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

//
// constructors and destructor
//
HLTElectronPixelMatchFilter::HLTElectronPixelMatchFilter(const edm::ParameterSet& iConfig)
{
  candTag_            = iConfig.getParameter< edm::InputTag > ("candTag");

  L1IsoPixelSeedsTag_  = iConfig.getParameter< edm::InputTag > ("L1IsoPixelSeedsTag");
  L1NonIsoPixelSeedsTag_  = iConfig.getParameter< edm::InputTag > ("L1NonIsoPixelSeedsTag");

  npixelmatchcut_     = iConfig.getParameter<double> ("npixelmatchcut");
  ncandcut_           = iConfig.getParameter<int> ("ncandcut");

  doIsolated_    = iConfig.getParameter<bool> ("doIsolated");

   store_ = iConfig.getUntrackedParameter<bool> ("SaveTag",false) ;
   L1IsoCollTag_= iConfig.getParameter< edm::InputTag > ("L1IsoCand"); 
   L1NonIsoCollTag_= iConfig.getParameter< edm::InputTag > ("L1NonIsoCand"); 

   //register your products
   produces<trigger::TriggerFilterObjectWithRefs>();
}


HLTElectronPixelMatchFilter::~HLTElectronPixelMatchFilter(){}


// ------------ method called to produce the data  ------------
bool
HLTElectronPixelMatchFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  // The filter object
  using namespace trigger;
    std::auto_ptr<trigger::TriggerFilterObjectWithRefs> filterproduct (new trigger::TriggerFilterObjectWithRefs(path(),module()));
    if( store_ ){filterproduct->addCollectionTag(L1IsoCollTag_);}
    if( store_ && !doIsolated_){filterproduct->addCollectionTag(L1NonIsoCollTag_);}
  // Ref to Candidate object to be recorded in filter object
   edm::Ref<reco::RecoEcalCandidateCollection> ref;


  edm::Handle<trigger::TriggerFilterObjectWithRefs> PrevFilterOutput;

  iEvent.getByLabel (candTag_,PrevFilterOutput);

  std::vector<edm::Ref<reco::RecoEcalCandidateCollection> > recoecalcands;
  PrevFilterOutput->getObjects(TriggerCluster, recoecalcands);

  //get hold of the pixel seed - supercluster association map
  edm::Handle<reco::ElectronPixelSeedCollection> L1IsoSeeds;
  iEvent.getByLabel (L1IsoPixelSeedsTag_,L1IsoSeeds);

  edm::Handle<reco::ElectronPixelSeedCollection> L1NonIsoSeeds;
  if(!doIsolated_){
    iEvent.getByLabel (L1NonIsoPixelSeedsTag_,L1NonIsoSeeds);
  }
  
  // look at all egammas,  check cuts and add to filter object
  int n = 0;

  for (unsigned int i=0; i<recoecalcands.size(); i++) {

    ref = recoecalcands[i];
    reco::SuperClusterRef recr2 = ref->superCluster();
    int nmatch = 0;

    for(reco::ElectronPixelSeedCollection::const_iterator it = L1IsoSeeds->begin(); 
	it != L1IsoSeeds->end(); it++){
      const reco::SuperClusterRef & scRef=it->superCluster();

      if(&(*recr2) ==  &(*scRef)) {
	nmatch++;
      }
    }
    
    if(!doIsolated_){

      for(reco::ElectronPixelSeedCollection::const_iterator it = L1NonIsoSeeds->begin(); 
	  it != L1NonIsoSeeds->end(); it++){
	const reco::SuperClusterRef & scRef=it->superCluster();
      
	if(&(*recr2) ==  &(*scRef)) {
	  nmatch++;
	}
      }

    }//end if(!doIsolated_)
    
    if ( nmatch >= npixelmatchcut_) {
      n++;
      filterproduct->addObject(TriggerCluster, ref);
    }
    
  }//end of loop over candidates
   
  // filter decision
  bool accept(n>=ncandcut_);
  
  // put filter object into the Event
  iEvent.put(filterproduct);
  
  return accept;
}

