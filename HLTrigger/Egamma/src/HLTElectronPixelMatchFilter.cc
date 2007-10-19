/** \class HLTElectronPixelMatchFilter
 *
 * $Id: HLTElectronPixelMatchFilter.cc,v 1.6 2007/10/16 14:37:37 ghezzi Exp $
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

  L1IsoPixelSeedsTag_  = iConfig.getParameter< edm::InputTag > ("L1IsoPixelSeedsTag");
  //L1IsoPixelmapendcapTag_  = iConfig.getParameter< edm::InputTag > ("L1IsoPixelmapendcapTag");

  L1NonIsoPixelSeedsTag_  = iConfig.getParameter< edm::InputTag > ("L1NonIsoPixelSeedsTag");
  // L1NonIsoPixelmapendcapTag_  = iConfig.getParameter< edm::InputTag > ("L1NonIsoPixelmapendcapTag");

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
  edm::Handle<reco::ElectronPixelSeedCollection> L1IsoSeeds;
  iEvent.getByLabel (L1IsoPixelSeedsTag_,L1IsoSeeds);
  
  //get hold of the pixel seed - supercluster association map
  //  edm::Handle<reco::ElectronPixelSeedCollection> L1IsoEndcapSeeds;
  //iEvent.getByLabel (L1IsoPixelmapendcapTag_,L1IsoEndcapSeeds);

  edm::Handle<reco::ElectronPixelSeedCollection> L1NonIsoSeeds;
  //edm::Handle<reco::ElectronPixelSeedCollection> L1NonIsoEndcapSeeds;
  if(!doIsolated_){
    iEvent.getByLabel (L1NonIsoPixelSeedsTag_,L1NonIsoSeeds);
    // iEvent.getByLabel (L1NonIsoPixelmapendcapTag_,L1NonIsoEndcapSeeds);
  }
  
  // look at all egammas,  check cuts and add to filter object
  int n = 0;

  for (unsigned int i=0; i<recoecalcands->size(); i++) {
    candref = recoecalcands->getParticleRef(i);
    reco::RecoEcalCandidateRef recr = candref.castTo<reco::RecoEcalCandidateRef>();
    reco::SuperClusterRef recr2 = recr->superCluster();
    int nmatch = 0;

    for(reco::ElectronPixelSeedCollection::const_iterator it = L1IsoSeeds->begin(); 
	it != L1IsoSeeds->end(); it++){
      const reco::SuperClusterRef & scRef=it->superCluster();

      if(&(*recr2) ==  &(*scRef)) {
	nmatch++;
      }
    }
    
//     for(reco::ElectronPixelSeedCollection::const_iterator ite = L1IsoEndcapSeeds->begin(); 
// 	ite != L1IsoEndcapSeeds->end(); ite++){
//        const reco::SuperClusterRef & scRef=ite->superCluster();
//       if(&(*recr2) ==  &(*scRef)) {
// 	nmatch++;
//       }
//     }

    if(!doIsolated_){

      for(reco::ElectronPixelSeedCollection::const_iterator it = L1NonIsoSeeds->begin(); 
	  it != L1NonIsoSeeds->end(); it++){
	const reco::SuperClusterRef & scRef=it->superCluster();
      
	if(&(*recr2) ==  &(*scRef)) {
	  nmatch++;
	}
      }

//       for(reco::ElectronPixelSeedCollection::const_iterator ite = L1NonIsoEndcapSeeds->begin(); 
// 	  ite != L1NonIsoEndcapSeeds->end(); ite++){
      
// 	const reco::SuperClusterRef & scRef=ite->superCluster();
      
// 	if(&(*recr2) ==  &(*scRef)) {
// 	  nmatch++;
// 	}
//       } 

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

