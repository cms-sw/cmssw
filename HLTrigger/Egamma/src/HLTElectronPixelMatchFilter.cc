/** \class HLTElectronPixelMatchFilter
 *
 *
 *  \author Monica Vazquez Acosta (CERN)
 *
 */

#include "HLTrigger/Egamma/interface/HLTElectronPixelMatchFilter.h"

#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Common/interface/AssociationMap.h"

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"

#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

//
// constructors and destructor
//
HLTElectronPixelMatchFilter::HLTElectronPixelMatchFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig)
{
  candTag_            = iConfig.getParameter< edm::InputTag > ("candTag");
  L1IsoPixelSeedsTag_  = iConfig.getParameter< edm::InputTag > ("L1IsoPixelSeedsTag");
  L1NonIsoPixelSeedsTag_  = iConfig.getParameter< edm::InputTag > ("L1NonIsoPixelSeedsTag");
  npixelmatchcut_     = iConfig.getParameter<double> ("npixelmatchcut");
  ncandcut_           = iConfig.getParameter<int> ("ncandcut");
  doIsolated_    = iConfig.getParameter<bool> ("doIsolated");
  L1IsoCollTag_= iConfig.getParameter< edm::InputTag > ("L1IsoCand");
  L1NonIsoCollTag_= iConfig.getParameter< edm::InputTag > ("L1NonIsoCand");

  candToken_ = consumes<trigger::TriggerFilterObjectWithRefs>(candTag_);
  L1IsoPixelSeedsToken_ = consumes<reco::ElectronSeedCollection>(L1IsoPixelSeedsTag_);
  L1NonIsoPixelSeedsToken_= consumes<reco::ElectronSeedCollection>(L1NonIsoPixelSeedsTag_);
}

HLTElectronPixelMatchFilter::~HLTElectronPixelMatchFilter(){}

void
HLTElectronPixelMatchFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("candTag",edm::InputTag("hltEgammaHcalIsolFilter"));
  desc.add<edm::InputTag>("L1IsoPixelSeedsTag",edm::InputTag("electronPixelSeeds"));
  desc.add<edm::InputTag>("L1NonIsoPixelSeedsTag",edm::InputTag("electronPixelSeeds"));
  desc.add<double>("npixelmatchcut",1.0);
  desc.add<int>("ncandcut",1);
  desc.add<bool>("doIsolated",true);
  desc.add<edm::InputTag>("L1IsoCand",edm::InputTag("hltL1IsoRecoEcalCandidate"));
  desc.add<edm::InputTag>("L1NonIsoCand",edm::InputTag("hltL1NonIsoRecoEcalCandidate"));
  descriptions.add("hltElectronPixelMatchFilter",desc);
}

// ------------ method called to produce the data  ------------
bool
HLTElectronPixelMatchFilter::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct) const
{
  // The filter object
  using namespace trigger;
  if (saveTags()) {
    filterproduct.addCollectionTag(L1IsoCollTag_);
    if (not doIsolated_) filterproduct.addCollectionTag(L1NonIsoCollTag_);
  }

  // Ref to Candidate object to be recorded in filter object
  edm::Ref<reco::RecoEcalCandidateCollection> ref;

  edm::Handle<trigger::TriggerFilterObjectWithRefs> PrevFilterOutput;
  iEvent.getByToken (candToken_,PrevFilterOutput);

  std::vector<edm::Ref<reco::RecoEcalCandidateCollection> > recoecalcands;
  PrevFilterOutput->getObjects(TriggerCluster, recoecalcands);
  if(recoecalcands.empty()) PrevFilterOutput->getObjects(TriggerPhoton,recoecalcands);  //we dont know if its type trigger cluster or trigger photon

  //get hold of the pixel seed - supercluster association map
  edm::Handle<reco::ElectronSeedCollection> L1IsoSeeds;
  iEvent.getByToken (L1IsoPixelSeedsToken_,L1IsoSeeds);

  edm::Handle<reco::ElectronSeedCollection> L1NonIsoSeeds;
  if(!doIsolated_){
    iEvent.getByToken (L1NonIsoPixelSeedsToken_,L1NonIsoSeeds);
  }

  // look at all egammas,  check cuts and add to filter object
  int n = 0;
  // std::cout<<"@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"<<std::endl;
  for (unsigned int i=0; i<recoecalcands.size(); i++) {

    ref = recoecalcands[i];
    reco::SuperClusterRef recr2 = ref->superCluster();

    // std::cout<<"AA  ref, e, eta, phi"<<&(*recr2)<<" "<<recr2->energy()<<" "<<recr2->eta()<<" "<<recr2->phi()<<std::endl;
    int nmatch = 0;

    for(reco::ElectronSeedCollection::const_iterator it = L1IsoSeeds->begin();
	it != L1IsoSeeds->end(); it++){

      edm::RefToBase<reco::CaloCluster> caloCluster = it->caloCluster() ;
      reco::SuperClusterRef scRef = caloCluster.castTo<reco::SuperClusterRef>() ;
      // std::cout<<"BB ref, e, eta, phi"<<&(*scRef)<<" "<<scRef->energy()<<" "<<scRef->eta()<<" "<<scRef->phi()<<std::endl;

      if(&(*recr2) ==  &(*scRef)) {
	nmatch++;
      }
    }

    if(!doIsolated_){

      for(reco::ElectronSeedCollection::const_iterator it = L1NonIsoSeeds->begin();
	  it != L1NonIsoSeeds->end(); it++){
	edm::RefToBase<reco::CaloCluster> caloCluster = it->caloCluster() ;
	reco::SuperClusterRef scRef = caloCluster.castTo<reco::SuperClusterRef>() ;
	//std::cout<<"CC ref, e, eta, phi"<<&(*scRef)<<" "<<scRef->energy()<<" "<<scRef->eta()<<" "<<scRef->phi()<<std::endl;
	if(&(*recr2) ==  &(*scRef)) {
	  nmatch++;
	}
      }

    }//end if(!doIsolated_)

    if ( nmatch >= npixelmatchcut_) {
      n++;
      filterproduct.addObject(TriggerCluster, ref);
    }

  }//end of loop over candidates
  // std::cout<<"######################################################################"<<std::endl;
  // filter decision
  bool accept(n>=ncandcut_);

  return accept;
}


