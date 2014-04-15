/** \class HLTElectronPixelMatchFilter
 *
 * $Id: HLTElectronPixelMatchFilter.cc,v 1.15 2012/03/06 10:13:59 sharper Exp $
 *
 *  \author Aidan Randle-Conde (ULB)
 *
 */

#include "HLTrigger/Egamma/interface/HLTElectronPixelMatchFilter.h"

#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/AssociationMap.h"

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"

#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

HLTElectronPixelMatchFilter::HLTElectronPixelMatchFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig) {
  candTag_                = iConfig.getParameter< edm::InputTag > ("candTag");
  L1IsoPixelSeedsTag_     = iConfig.getParameter< edm::InputTag > ("L1IsoPixelSeedsTag");
  L1NonIsoPixelSeedsTag_  = iConfig.getParameter< edm::InputTag > ("L1NonIsoPixelSeedsTag");
  npixelmatchcut_         = iConfig.getParameter< double >        ("npixelmatchcut");
  ncandcut_               = iConfig.getParameter< int >           ("ncandcut");
  doIsolated_             = iConfig.getParameter< bool >          ("doIsolated");
  L1IsoCollTag_           = iConfig.getParameter< edm::InputTag > ("L1IsoCand");
  L1NonIsoCollTag_        = iConfig.getParameter< edm::InputTag > ("L1NonIsoCand");
  
  candToken_ = consumes<trigger::TriggerFilterObjectWithRefs>(candTag_);
  L1IsoPixelSeedsToken_ = consumes<reco::ElectronSeedCollection>(L1IsoPixelSeedsTag_);
  L1NonIsoPixelSeedsToken_= consumes<reco::ElectronSeedCollection>(L1NonIsoPixelSeedsTag_);
  
  s_a_phi1B_ = iConfig.getParameter< double >("s_a_phi1B") ;
  s_a_phi1I_ = iConfig.getParameter< double >("s_a_phi1I") ;
  s_a_phi1F_ = iConfig.getParameter< double >("s_a_phi1F") ;
  s_a_phi2B_ = iConfig.getParameter< double >("s_a_phi2B") ;
  s_a_phi2I_ = iConfig.getParameter< double >("s_a_phi2I") ;
  s_a_phi2F_ = iConfig.getParameter< double >("s_a_phi2F") ;
  s_a_zB_    = iConfig.getParameter< double >("s_a_zB"   ) ;
  s_a_rI_    = iConfig.getParameter< double >("s_a_rI"   ) ;
  s_a_rF_    = iConfig.getParameter< double >("s_a_rF"   ) ;
  s2_threshold_ = iConfig.getParameter< double >("s2_threshold") ;
  useS_      = iConfig.getParameter< bool >("useS" );
  
  s_b_phi1B_ = 1.0/s_a_phi1B_ ;
  s_b_phi1I_ = 1.0/s_a_phi1I_ ;
  s_b_phi1F_ = 1.0/s_a_phi1F_ ;
  s_b_phi2B_ = 1.0/s_a_phi2B_ ;
  s_b_phi2I_ = 1.0/s_a_phi2I_ ;
  s_b_phi2F_ = 1.0/s_a_phi2F_ ;
  s_b_zB_    = 1.0/s_a_zB_    ;
  s_b_rI_    = 1.0/s_a_rI_    ;
  s_b_rF_    = 1.0/s_a_rF_    ;
}

HLTElectronPixelMatchFilter::~HLTElectronPixelMatchFilter()
{}

float HLTElectronPixelMatchFilter::calculate_s2(reco::ElectronSeedCollection::const_iterator it, int charge) const {
  int subDet1 = it->subDet1() ;
  int subDet2 = it->subDet2() ;
  if(charge<0){ // Negative
    if(subDet1==1 && subDet2==1){ // Barrel
      return pow(s_b_phi1B_*it->dPhi1(),2) + pow(s_b_phi2B_*it->dPhi2(),2) +pow(s_b_zB_*it->dRz1(),2) ;
    }
    else if(subDet1==1 && subDet2!=1){ // Intermediate
      return pow(s_b_phi1I_*it->dPhi1(),2) + pow(s_b_phi1I_*it->dPhi2(),2) +pow(s_b_rI_*it->dRz1(),2) ;
    }
    else if(subDet1!=1 && subDet2!=1){ // Forward
      return pow(s_b_phi1F_*it->dPhi1(),2) + pow(s_b_phi1F_*it->dPhi2(),2) +pow(s_b_rF_*it->dRz1(),2) ;
    }
  }
  else{ // Positive
    if(subDet1==1 && subDet2==1){ // Barrel
      return pow(s_b_phi1B_*it->dPhi1(),2) + pow(s_b_phi1B_*it->dPhi2(),2) +pow(s_b_zB_*it->dRz1Pos(),2) ;
    }
    else if(subDet1==1 && subDet2!=1){ // Intermediate
      return pow(s_b_phi1I_*it->dPhi1(),2) + pow(s_b_phi1I_*it->dPhi2(),2) +pow(s_b_rI_*it->dRz1Pos(),2) ;
    }
    else if((subDet1=!1) && (subDet2!=1)){ // Forward
      return pow(s_b_phi1F_*it->dPhi1(),2) + pow(s_b_phi1F_*it->dPhi2(),2) +pow(s_b_rF_*it->dRz1Pos(),2) ;
    }
  }

  return 999 ;
}

void HLTElectronPixelMatchFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
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
  desc.add<double>("s_a_phi1B",    0.0069) ;
  desc.add<double>("s_a_phi1I",    0.0088) ;
  desc.add<double>("s_a_phi1F",    0.0076) ;
  desc.add<double>("s_a_phi2B",    0.00037) ;
  desc.add<double>("s_a_phi2I",    0.00070) ;
  desc.add<double>("s_a_phi2F",    0.00906) ;
  desc.add<double>("s_a_zB"   ,    0.012) ;
  desc.add<double>("s_a_rI"   ,    0.027) ;
  desc.add<double>("s_a_rF"   ,    0.040) ;
  desc.add<double>("s2_threshold", 0.4) ;
  desc.add<bool>  ("useS"     , false);
  descriptions.add("hltElectronPixelMatchFilter",desc);
}

bool HLTElectronPixelMatchFilter::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct) const {
  // The filter object
  using namespace trigger;
  if (saveTags()) {
    filterproduct.addCollectionTag(L1IsoCollTag_);
    if (not doIsolated_) filterproduct.addCollectionTag(L1NonIsoCollTag_);
  }
  
  // Ref to Candidate object to be recorded in filter object
  edm::Ref<reco::RecoEcalCandidateCollection> ref;
  
  edm::Handle<trigger::TriggerFilterObjectWithRefs> PrevFilterOutput;
  iEvent.getByToken(candToken_,PrevFilterOutput);
  
  std::vector<edm::Ref<reco::RecoEcalCandidateCollection> > recoecalcands;
  PrevFilterOutput->getObjects(TriggerCluster, recoecalcands);
  if(recoecalcands.empty()) PrevFilterOutput->getObjects(TriggerPhoton,recoecalcands);  //we dont know if its type trigger cluster or trigger photon
  
  //get hold of the pixel seed - supercluster association map
  edm::Handle<reco::ElectronSeedCollection> L1IsoSeeds;
  iEvent.getByToken(L1IsoPixelSeedsToken_,L1IsoSeeds);

  edm::Handle<reco::ElectronSeedCollection> L1NonIsoSeeds;
  if(!doIsolated_){
    iEvent.getByToken(L1NonIsoPixelSeedsToken_,L1NonIsoSeeds);
  }
  
  // look at all egammas,  check cuts and add to filter object
  int n = 0;
  for (unsigned int i=0; i<recoecalcands.size(); i++) {

    ref = recoecalcands[i];
    reco::SuperClusterRef recr2 = ref->superCluster();

    int nmatch = 0;
    
    float el_best_s2_tmp = 1e6 ;
    for(reco::ElectronSeedCollection::const_iterator it = L1IsoSeeds->begin(); it != L1IsoSeeds->end(); it++){
      edm::RefToBase<reco::CaloCluster> caloCluster = it->caloCluster() ;
      reco::SuperClusterRef scRef = caloCluster.castTo<reco::SuperClusterRef>() ;
      if(&(*recr2) ==  &(*scRef)){
        if(useS_){
          float el_s2_neg = calculate_s2(it,-1) ;
          float el_s2_pos = calculate_s2(it, 1) ;
          if(el_s2_neg<el_best_s2_tmp){
            el_best_s2_tmp = el_s2_neg ;
          }
          if(el_s2_pos<el_best_s2_tmp){
            el_best_s2_tmp = el_s2_pos ;
          }
          if(el_s2_neg<s2_threshold_ || el_s2_pos<s2_threshold_) nmatch++ ;
        }
        else{
          nmatch++;
        }
      }
    }

    if(!doIsolated_){
      for(reco::ElectronSeedCollection::const_iterator it = L1NonIsoSeeds->begin(); it != L1NonIsoSeeds->end(); it++){
        edm::RefToBase<reco::CaloCluster> caloCluster = it->caloCluster() ;
        reco::SuperClusterRef scRef = caloCluster.castTo<reco::SuperClusterRef>() ;
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
  
  // filter decision
  bool accept(n>=ncandcut_);
  accept = true ;
  
  return accept;
}


