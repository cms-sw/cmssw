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
  candTag_           = iConfig.getParameter< edm::InputTag > ("candTag");
  l1PixelSeedsTag_   = iConfig.getParameter< edm::InputTag > ("l1PixelSeedsTag");
  npixelmatchcut_    = iConfig.getParameter< double >        ("npixelmatchcut");
  ncandcut_          = iConfig.getParameter< int >           ("ncandcut");
  l1EGTag_           = iConfig.getParameter< edm::InputTag > ("l1EGCand");
  
  candToken_         = consumes<trigger::TriggerFilterObjectWithRefs> (candTag_);
  l1PixelSeedsToken_ = consumes<reco::ElectronSeedCollection> (l1PixelSeedsTag_);
  
  sPhi1B_         = iConfig.getParameter< double >("s_a_phi1B") ;
  sPhi1I_         = iConfig.getParameter< double >("s_a_phi1I") ;
  sPhi1F_         = iConfig.getParameter< double >("s_a_phi1F") ;
  sPhi2B_         = iConfig.getParameter< double >("s_a_phi2B") ;
  sPhi2I_         = iConfig.getParameter< double >("s_a_phi2I") ;
  sPhi2F_         = iConfig.getParameter< double >("s_a_phi2F") ;
  sZ2B_           = iConfig.getParameter< double >("s_a_zB"   ) ;
  sR2I_           = iConfig.getParameter< double >("s_a_rI"   ) ;
  sR2F_           = iConfig.getParameter< double >("s_a_rF"   ) ;
  s2BarrelThres_  = std::pow(std::atanh(iConfig.getParameter< double >("tanhSO10BarrelThres"))*10., 2);
  s2InterThres_   = std::pow(std::atanh(iConfig.getParameter< double >("tanhSO10InterThres"))*10., 2);
  s2ForwardThres_ = std::pow(std::atanh(iConfig.getParameter< double >("tanhSO10ForwardThres"))*10., 2);
  
  isPixelVeto_ = iConfig.getParameter< bool >("pixelVeto");
  useS_        = iConfig.getParameter< bool >("useS" );

}

HLTElectronPixelMatchFilter::~HLTElectronPixelMatchFilter()
{}

float HLTElectronPixelMatchFilter::calDPhi1Sq(reco::ElectronSeedCollection::const_iterator seed, int charge)const
{
  const float dPhi1Const = seed->subDet1()==1 ? seed->subDet2()==1 ? sPhi1B_ : sPhi1I_ : sPhi1F_;
  float dPhi1 = charge<0 ? seed->dPhi1()/dPhi1Const :  seed->dPhi1Pos()/dPhi1Const; 
  return dPhi1*dPhi1;
}

float HLTElectronPixelMatchFilter::calDPhi2Sq(reco::ElectronSeedCollection::const_iterator seed, int charge)const
{
  const float dPhi2Const = seed->subDet1()==1 ? seed->subDet2()==1 ? sPhi2B_ : sPhi2I_ : sPhi2F_;
  float dPhi2 = charge <0 ? seed->dPhi2()/dPhi2Const :  seed->dPhi2Pos()/dPhi2Const;
  return dPhi2*dPhi2;
}


float HLTElectronPixelMatchFilter::calDZ2Sq(reco::ElectronSeedCollection::const_iterator seed, int charge)const
{
  const float dRZ2Const = seed->subDet1()==1 ? seed->subDet2()==1 ? sZ2B_ : sR2I_ : sR2F_;
  float dRZ2 = charge<0 ? seed->dRz2()/dRZ2Const : seed->dRz2Pos()/dRZ2Const;
  return dRZ2*dRZ2;
}

void HLTElectronPixelMatchFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("candTag", edm::InputTag("hltEgammaHcalIsolFilter"));
  desc.add<edm::InputTag>("l1PixelSeedsTag", edm::InputTag("electronPixelSeeds"));
  desc.add<double>("npixelmatchcut", 1.0);
  desc.add<int>("ncandcut", 1);
  desc.add<edm::InputTag>("l1EGCand", edm::InputTag("hltL1IsoRecoEcalCandidate"));
  desc.add<double>("s_a_phi1B",    0.0069) ;
  desc.add<double>("s_a_phi1I",    0.0088) ;
  desc.add<double>("s_a_phi1F",    0.0076) ;
  desc.add<double>("s_a_phi2B",    0.00037) ;
  desc.add<double>("s_a_phi2I",    0.00070) ;
  desc.add<double>("s_a_phi2F",    0.00906) ;
  desc.add<double>("s_a_zB"   ,    0.012) ;
  desc.add<double>("s_a_rI"   ,    0.027) ;
  desc.add<double>("s_a_rF"   ,    0.040) ;
  desc.add<double>("s2_threshold", 0);
  desc.add<double>("tanhSO10BarrelThres", 0.35);
  desc.add<double>("tanhSO10InterThres", 1);
  desc.add<double>("tanhSO10ForwardThres", 1);
  desc.add<bool>  ("useS"     , false);
  desc.add<bool>  ("pixelVeto", false);

  descriptions.add("hltElectronPixelMatchFilter",desc);
}

bool HLTElectronPixelMatchFilter::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct) const {
  // The filter object
  using namespace trigger;
  if (saveTags()) {
    filterproduct.addCollectionTag(l1EGTag_);
  }
  
  // Ref to Candidate object to be recorded in filter object
  edm::Ref<reco::RecoEcalCandidateCollection> ref;
  
  edm::Handle<trigger::TriggerFilterObjectWithRefs> PrevFilterOutput;
  iEvent.getByToken(candToken_,PrevFilterOutput);
  
  std::vector<edm::Ref<reco::RecoEcalCandidateCollection> > recoecalcands;
  PrevFilterOutput->getObjects(TriggerCluster, recoecalcands);
  if(recoecalcands.empty()) PrevFilterOutput->getObjects(TriggerPhoton,recoecalcands);  //we dont know if its type trigger cluster or trigger photon
  
  //get hold of the pixel seed - supercluster association map
  edm::Handle<reco::ElectronSeedCollection> l1PixelSeeds;
  iEvent.getByToken(l1PixelSeedsToken_, l1PixelSeeds);
  
  // look at all egammas,  check cuts and add to filter object
  int n = 0;
  for (unsigned int i=0; i<recoecalcands.size(); i++) {

    ref = recoecalcands[i];
    reco::SuperClusterRef recr2 = ref->superCluster();
    
    int nmatch = getNrOfMatches(l1PixelSeeds, recr2);

    if (!isPixelVeto_) {
      if ( nmatch >= npixelmatchcut_) {
	n++;
	filterproduct.addObject(TriggerCluster, ref);
      }
    } else {
      if ( nmatch == 0) {
	n++;
	filterproduct.addObject(TriggerCluster, ref);
      }
    }
 
  }//end of loop over candidates
  
  // filter decision
  const bool accept(n>=ncandcut_);
  return accept;
}

int HLTElectronPixelMatchFilter::getNrOfMatches(edm::Handle<reco::ElectronSeedCollection>& eleSeeds,
						reco::SuperClusterRef& candSCRef)const
{
  int nrMatch=0;
  for(reco::ElectronSeedCollection::const_iterator seedIt = eleSeeds->begin(); seedIt != eleSeeds->end(); seedIt++){
    edm::RefToBase<reco::CaloCluster> caloCluster = seedIt->caloCluster() ;
    reco::SuperClusterRef scRef = caloCluster.castTo<reco::SuperClusterRef>() ;
    if(&(*candSCRef) ==  &(*scRef)){
      if(useS_){
	float s2Neg = calDPhi1Sq(seedIt,-1) + calDPhi2Sq(seedIt,-1) + calDZ2Sq(seedIt,-1);
	float s2Pos = calDPhi1Sq(seedIt,1) + calDPhi2Sq(seedIt,1) + calDZ2Sq(seedIt,1);
	
	const float s2Thres = seedIt->subDet1()==1 ? seedIt->subDet2()==1 ? s2BarrelThres_ : s2InterThres_ : s2ForwardThres_; 
	if(s2Neg<s2Thres || s2Pos<s2Thres) nrMatch++ ;
      }
      else nrMatch++;
    }//end sc ref match
  }//end loop over ele seeds
  return nrMatch;
}
