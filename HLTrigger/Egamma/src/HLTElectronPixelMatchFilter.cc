/** \class HLTElectronPixelMatchFilter
 *
 * $Id: HLTElectronPixelMatchFilter.cc,v 1.4 2007/05/08 13:22:44 ghezzi Exp $
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

#include "DataFormats/EgammaReco/interface/ElectronPixelSeed.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SeedSuperClusterAssociation.h"

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
  edm::Handle<reco::SeedSuperClusterAssociationCollection> L1IsoBarrelMap;
  iEvent.getByLabel (L1IsoPixelmapbarrelTag_,L1IsoBarrelMap);
  
  //get hold of the pixel seed - supercluster association map
  edm::Handle<reco::SeedSuperClusterAssociationCollection> L1IsoEndcapMap;
  iEvent.getByLabel (L1IsoPixelmapendcapTag_,L1IsoEndcapMap);

  edm::Handle<reco::SeedSuperClusterAssociationCollection> L1NonIsoBarrelMap;
  edm::Handle<reco::SeedSuperClusterAssociationCollection> L1NonIsoEndcapMap;
  if(!doIsolated_){
    iEvent.getByLabel (L1NonIsoPixelmapbarrelTag_,L1NonIsoBarrelMap);
    iEvent.getByLabel (L1NonIsoPixelmapendcapTag_,L1NonIsoEndcapMap);
  }
  
  // look at all egammas,  check cuts and add to filter object
  int n = 0;

  for (unsigned int i=0; i<recoecalcands->size(); i++) {
    candref = recoecalcands->getParticleRef(i);
    reco::RecoEcalCandidateRef recr = candref.castTo<reco::RecoEcalCandidateRef>();
    reco::SuperClusterRef recr2 = recr->superCluster();

    int nmatch = 0;

    for(reco::SeedSuperClusterAssociationCollection::const_iterator itb = L1IsoBarrelMap->begin(); 
	itb != L1IsoBarrelMap->end(); itb++){
      
      edm::Ref<reco::SuperClusterCollection> theClusBarrel = itb->val;
      
      if(&(*recr2) ==  &(*theClusBarrel)) {
	nmatch++;
      }
    }

    for(reco::SeedSuperClusterAssociationCollection::const_iterator ite = L1IsoEndcapMap->begin(); 
	ite != L1IsoEndcapMap->end(); ite++){
      
      edm::Ref<reco::SuperClusterCollection> theClusEndcap = ite->val;
      
      if(&(*recr2) ==  &(*theClusEndcap)) {
	nmatch++;
      }
    }

    if(!doIsolated_){

      for(reco::SeedSuperClusterAssociationCollection::const_iterator itb = L1NonIsoBarrelMap->begin(); 
	  itb != L1NonIsoBarrelMap->end(); itb++){
      
	edm::Ref<reco::SuperClusterCollection> theClusBarrel = itb->val;
      
	if(&(*recr2) ==  &(*theClusBarrel)) {
	  nmatch++;
	}
      }

      for(reco::SeedSuperClusterAssociationCollection::const_iterator ite = L1NonIsoEndcapMap->begin(); 
	  ite != L1NonIsoEndcapMap->end(); ite++){
      
	edm::Ref<reco::SuperClusterCollection> theClusEndcap = ite->val;
      
	if(&(*recr2) ==  &(*theClusEndcap)) {
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

