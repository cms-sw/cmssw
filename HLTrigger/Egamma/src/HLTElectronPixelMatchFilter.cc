/** \class HLTElectronPixelMatchFilter
 *
 * $Id: HLTElectronPixelMatchFilter.cc,v 1.5 2007/09/20 00:05:22 ratnik Exp $
 *
 *  \author Monica Vazquez Acosta (CERN)
 *
 */

#include "HLTrigger/Egamma/interface/HLTElectronPixelMatchFilter.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/HLTReco/interface/HLTFilterObject.h"

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

  L1IsoPixelmapbarrelTag_  = iConfig.getParameter< edm::InputTag > ("L1IsoPixelmapbarrelTag");
  L1IsoPixelmapendcapTag_  = iConfig.getParameter< edm::InputTag > ("L1IsoPixelmapendcapTag");

  L1NonIsoPixelmapbarrelTag_  = iConfig.getParameter< edm::InputTag > ("L1NonIsoPixelmapbarrelTag");
  L1NonIsoPixelmapendcapTag_  = iConfig.getParameter< edm::InputTag > ("L1NonIsoPixelmapendcapTag");

  npixelmatchcut_     = iConfig.getParameter<double> ("npixelmatchcut");
  ncandcut_           = iConfig.getParameter<int> ("ncandcut");

  doIsolated_    = iConfig.getParameter<bool> ("doIsolated");

   //register your products
   produces<reco::HLTFilterObjectWithRefs>();
}

HLTElectronPixelMatchFilter::~HLTElectronPixelMatchFilter(){}


// ------------ method called to produce the data  ------------
bool
HLTElectronPixelMatchFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  // The filter object
  std::auto_ptr<reco::HLTFilterObjectWithRefs> filterproduct (new reco::HLTFilterObjectWithRefs(path(),module()));
  // Ref to Candidate object to be recorded in filter object
  edm::RefToBase<reco::Candidate> candref;
  
  // get hold of filtered candidates
  edm::Handle<reco::HLTFilterObjectWithRefs> recoecalcands;
  iEvent.getByLabel (candTag_,recoecalcands);
  
  //get hold of the pixel seed - supercluster association map
  edm::Handle<reco::ElectronPixelSeedCollection> L1IsoBarrelSeeds;
  iEvent.getByLabel (L1IsoPixelmapbarrelTag_,L1IsoBarrelSeeds);
  
  //get hold of the pixel seed - supercluster association map
  edm::Handle<reco::ElectronPixelSeedCollection> L1IsoEndcapSeeds;
  iEvent.getByLabel (L1IsoPixelmapendcapTag_,L1IsoEndcapSeeds);

  edm::Handle<reco::ElectronPixelSeedCollection> L1NonIsoBarrelSeeds;
  edm::Handle<reco::ElectronPixelSeedCollection> L1NonIsoEndcapSeeds;
  if(!doIsolated_){
    iEvent.getByLabel (L1NonIsoPixelmapbarrelTag_,L1NonIsoBarrelSeeds);
    iEvent.getByLabel (L1NonIsoPixelmapendcapTag_,L1NonIsoEndcapSeeds);
  }
  
  // look at all egammas,  check cuts and add to filter object
  int n = 0;

  for (unsigned int i=0; i<recoecalcands->size(); i++) {
    candref = recoecalcands->getParticleRef(i);
    reco::RecoEcalCandidateRef recr = candref.castTo<reco::RecoEcalCandidateRef>();
    reco::SuperClusterRef recr2 = recr->superCluster();
    int nmatch = 0;

    for(reco::ElectronPixelSeedCollection::const_iterator itb = L1IsoBarrelSeeds->begin(); 
	itb != L1IsoBarrelSeeds->end(); itb++){
      const reco::SuperClusterRef & scRef=itb->superCluster();

      if(&(*recr2) ==  &(*scRef)) {
	nmatch++;
      }
    }
    
    for(reco::ElectronPixelSeedCollection::const_iterator ite = L1IsoEndcapSeeds->begin(); 
	ite != L1IsoEndcapSeeds->end(); ite++){
       const reco::SuperClusterRef & scRef=ite->superCluster();
      if(&(*recr2) ==  &(*scRef)) {
	nmatch++;
      }
    }

    if(!doIsolated_){

      for(reco::ElectronPixelSeedCollection::const_iterator itb = L1NonIsoBarrelSeeds->begin(); 
	  itb != L1NonIsoBarrelSeeds->end(); itb++){
	const reco::SuperClusterRef & scRef=itb->superCluster();
      
	if(&(*recr2) ==  &(*scRef)) {
	  nmatch++;
	}
      }

      for(reco::ElectronPixelSeedCollection::const_iterator ite = L1NonIsoEndcapSeeds->begin(); 
	  ite != L1NonIsoEndcapSeeds->end(); ite++){
      
	const reco::SuperClusterRef & scRef=ite->superCluster();
      
	if(&(*recr2) ==  &(*scRef)) {
	  nmatch++;
	}
      } 

    }

    if ( nmatch >= npixelmatchcut_) {
      n++;
      filterproduct->putParticle(candref);
    }
    
  }
   
  // filter decision
  bool accept(n>=ncandcut_);
  
  // put filter object into the Event
  iEvent.put(filterproduct);
  
  return accept;
}

