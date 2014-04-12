/** \class HLTEgammaL1MatchFilterPairs
 *
 *
 *  \author Monica Vazquez Acosta (CERN)
 *
 */

#include "HLTrigger/Egamma/interface/HLTEgammaL1MatchFilterPairs.h"

//#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/L1TObjects/interface/L1CaloGeometry.h"
#include "CondFormats/DataRecord/interface/L1CaloGeometryRecord.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include <vector>
#include <cmath>

#define TWOPI 2*M_PI
//
// constructors and destructor
//
HLTEgammaL1MatchFilterPairs::HLTEgammaL1MatchFilterPairs(const edm::ParameterSet& iConfig) : HLTFilter(iConfig)
{
   candIsolatedTag_ = iConfig.getParameter< edm::InputTag > ("candIsolatedTag");
   l1IsolatedTag_ = iConfig.getParameter< edm::InputTag > ("l1IsolatedTag");
   candNonIsolatedTag_ = iConfig.getParameter< edm::InputTag > ("candNonIsolatedTag");
   l1NonIsolatedTag_ = iConfig.getParameter< edm::InputTag > ("l1NonIsolatedTag");
   L1SeedFilterTag_ = iConfig.getParameter< edm::InputTag > ("L1SeedFilterTag");

   AlsoNonIsolatedFirst_ = iConfig.getParameter<bool>("AlsoNonIsolatedFirst");
   AlsoNonIsolatedSecond_  = iConfig.getParameter<bool>("AlsoNonIsolatedSecond");

   region_eta_size_      = iConfig.getParameter<double> ("region_eta_size");
   region_eta_size_ecap_ = iConfig.getParameter<double> ("region_eta_size_ecap");
   region_phi_size_      = iConfig.getParameter<double> ("region_phi_size");
   barrel_end_           = iConfig.getParameter<double> ("barrel_end");
   endcap_end_           = iConfig.getParameter<double> ("endcap_end");

   candIsolatedToken_ = consumes<reco::RecoEcalCandidateCollection>(candIsolatedTag_);
   candNonIsolatedToken_ = consumes<reco::RecoEcalCandidateCollection>(candNonIsolatedTag_);
   L1SeedFilterToken_ = consumes<trigger::TriggerFilterObjectWithRefs>(L1SeedFilterTag_);
}

HLTEgammaL1MatchFilterPairs::~HLTEgammaL1MatchFilterPairs(){}

void
HLTEgammaL1MatchFilterPairs::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("candIsolatedTag",edm::InputTag("hltRecoIsolatedEcalCandidate"));
  desc.add<edm::InputTag>("l1IsolatedTag",edm::InputTag("l1extraParticles","Isolated"));
  desc.add<edm::InputTag>("candNonIsolatedTag",edm::InputTag("hltRecoNonIsolatedEcalCandidate"));
  desc.add<edm::InputTag>("l1NonIsolatedTag",edm::InputTag("l1extraParticles","NonIsolated"));
  desc.add<edm::InputTag>("L1SeedFilterTag",edm::InputTag("theL1SeedFilter"));
  desc.add<bool>("AlsoNonIsolatedFirst",false);
  desc.add<bool>("AlsoNonIsolatedSecond",false);
  desc.add<double>("region_eta_size",0.522);
  desc.add<double>("region_eta_size_ecap",1.0);
  desc.add<double>("region_phi_size",1.044);
  desc.add<double>("barrel_end",1.4791);
  desc.add<double>("endcap_end",2.65);
  descriptions.add("hltEgammaL1MatchFilterPairs",desc);
}



// ------------ method called to produce the data  ------------
bool
HLTEgammaL1MatchFilterPairs::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct) const
{

  using namespace trigger;
  using namespace l1extra;
  std::vector < std::pair< edm::Ref<reco::RecoEcalCandidateCollection>, edm::Ref<reco::RecoEcalCandidateCollection> > > thePairs;

  edm::Handle<reco::RecoEcalCandidateCollection> recoIsolecalcands;
  iEvent.getByToken(candIsolatedToken_,recoIsolecalcands);
  edm::Handle<reco::RecoEcalCandidateCollection> recoNonIsolecalcands;
  iEvent.getByToken(candNonIsolatedToken_,recoNonIsolecalcands);

  // create pairs <L1Iso,L1Iso> and optionally <L1Iso, L1NonIso>
   for (reco::RecoEcalCandidateCollection::const_iterator recoecalcand1= recoIsolecalcands->begin(); recoecalcand1!=recoIsolecalcands->end(); recoecalcand1++) {
     edm::Ref<reco::RecoEcalCandidateCollection> ref1 =  edm::Ref<reco::RecoEcalCandidateCollection>(recoIsolecalcands, distance(recoIsolecalcands->begin(),recoecalcand1) );
       for (reco::RecoEcalCandidateCollection::const_iterator recoecalcand2= recoIsolecalcands->begin(); recoecalcand2!=recoIsolecalcands->end(); recoecalcand2++) {
	 edm::Ref<reco::RecoEcalCandidateCollection> ref2 =  edm::Ref<reco::RecoEcalCandidateCollection>(recoIsolecalcands, distance(recoIsolecalcands->begin(),recoecalcand2) );
	 if( &(*ref1) != &(*ref2) ) {thePairs.push_back(std::pair< edm::Ref<reco::RecoEcalCandidateCollection>, edm::Ref<reco::RecoEcalCandidateCollection> > (ref1,ref2) );}
       }
       if (AlsoNonIsolatedSecond_){
	 for (reco::RecoEcalCandidateCollection::const_iterator recoecalcand2= recoNonIsolecalcands->begin(); recoecalcand2!=recoNonIsolecalcands->end(); recoecalcand2++) {
	   edm::Ref<reco::RecoEcalCandidateCollection> ref2 =  edm::Ref<reco::RecoEcalCandidateCollection>(recoNonIsolecalcands, distance(recoNonIsolecalcands->begin(),recoecalcand2) );
	   if( &(*ref1) != &(*ref2) ) {thePairs.push_back(std::pair< edm::Ref<reco::RecoEcalCandidateCollection>, edm::Ref<reco::RecoEcalCandidateCollection> > (ref1,ref2) );}
	 }
       }
   }


  // create pairs <L1NonIso,L1Iso> and optionally <L1NonIso, L1NonIso>
   if (AlsoNonIsolatedFirst_){
     for (reco::RecoEcalCandidateCollection::const_iterator recoecalcand1= recoNonIsolecalcands->begin(); recoecalcand1!=recoNonIsolecalcands->end(); recoecalcand1++) {
       edm::Ref<reco::RecoEcalCandidateCollection> ref1 =  edm::Ref<reco::RecoEcalCandidateCollection>(recoNonIsolecalcands, distance(recoNonIsolecalcands->begin(),recoecalcand1) );
       for (reco::RecoEcalCandidateCollection::const_iterator recoecalcand2= recoIsolecalcands->begin(); recoecalcand2!=recoIsolecalcands->end(); recoecalcand2++) {
	 edm::Ref<reco::RecoEcalCandidateCollection> ref2 =  edm::Ref<reco::RecoEcalCandidateCollection>(recoIsolecalcands, distance(recoIsolecalcands->begin(),recoecalcand2) );
	 if( &(*ref1) != &(*ref2) ) {thePairs.push_back(std::pair< edm::Ref<reco::RecoEcalCandidateCollection>, edm::Ref<reco::RecoEcalCandidateCollection> > (ref1,ref2) );}
       }
       if (AlsoNonIsolatedSecond_){
	 for (reco::RecoEcalCandidateCollection::const_iterator recoecalcand2= recoNonIsolecalcands->begin(); recoecalcand2!=recoNonIsolecalcands->end(); recoecalcand2++) {
	   edm::Ref<reco::RecoEcalCandidateCollection> ref2 =  edm::Ref<reco::RecoEcalCandidateCollection>(recoNonIsolecalcands, distance(recoNonIsolecalcands->begin(),recoecalcand2) );
	   if( &(*ref1) != &(*ref2) ) {thePairs.push_back(std::pair< edm::Ref<reco::RecoEcalCandidateCollection>, edm::Ref<reco::RecoEcalCandidateCollection> > (ref1,ref2) );}
	 }
       }
     }
   }


   // Get the CaloGeometry
  edm::ESHandle<L1CaloGeometry> l1CaloGeom ;
  iSetup.get<L1CaloGeometryRecord>().get(l1CaloGeom) ;


  // look at all candidates,  check cuts and add to filter object
  int n(0);

  edm::Handle<trigger::TriggerFilterObjectWithRefs> L1SeedOutput;
  iEvent.getByToken (L1SeedFilterToken_,L1SeedOutput);

  std::vector<l1extra::L1EmParticleRef > l1EGIso;
  L1SeedOutput->getObjects(TriggerL1IsoEG, l1EGIso);
  std::vector<l1extra::L1EmParticleRef > l1EGNonIso;
  L1SeedOutput->getObjects(TriggerL1NoIsoEG, l1EGNonIso);

//   std::cout<<"L1EGIso size: "<<l1EGIso.size()<<std::endl;
//   for (unsigned int i=0; i<l1EGIso.size(); i++){std::cout<<"L1EGIso Et Eta phi: "<<l1EGIso[i]->et()<<" "<<l1EGIso[i]->eta()<<" "<<l1EGIso[i]->phi()<<std::endl;}
//   std::cout<<"L1EGNonIso size: "<<l1EGNonIso.size()<<std::endl;
//   for (unsigned int i=0; i<l1EGNonIso.size(); i++){std::cout<<"L1EGNonIso Et Eta phi: "<<l1EGNonIso[i]->et()<<" "<<l1EGNonIso[i]->eta()<<" "<<l1EGNonIso[i]->phi()<<std::endl;}
//   std::cout<<"Lpair vector size: "<<thePairs.size()<<std::endl;

  std::vector < std::pair< edm::Ref<reco::RecoEcalCandidateCollection>, edm::Ref<reco::RecoEcalCandidateCollection> > >::iterator  pairsIt;
  for (pairsIt = thePairs.begin(); pairsIt != thePairs.end(); pairsIt++){
//      edm::Ref<reco::RecoEcalCandidateCollection> r1 = pairsIt->first;
//      edm::Ref<reco::RecoEcalCandidateCollection> r2 = pairsIt->second;
//      std::cout<<"1) Et Eta phi: "<<r1->et()<<" "<<r1->eta()<<" "<<r1->phi()<<" 2) Et eta phi: "<<r2->et()<<" "<<r2->eta()<<" "<<r2->phi()<<std::endl;

    if ( CheckL1Matching(pairsIt->first,l1EGIso,l1EGNonIso) && CheckL1Matching(pairsIt->second,l1EGIso,l1EGNonIso) ){
      filterproduct.addObject(TriggerCluster, pairsIt->first);
      filterproduct.addObject(TriggerCluster, pairsIt->second);
      n++;
    }
  }


  //  std::cout<<"#####################################################"<<std::endl;
  // filter decision
  bool accept(n>=1);

  return accept;
}

bool HLTEgammaL1MatchFilterPairs::CheckL1Matching(edm::Ref<reco::RecoEcalCandidateCollection> ref, std::vector<l1extra::L1EmParticleRef >& l1EGIso, std::vector<l1extra::L1EmParticleRef >& l1EGNonIso) const {

  for (unsigned int i=0; i<l1EGIso.size(); i++) {
    //ORCA matching method
    double etaBinLow  = 0.;
    double etaBinHigh = 0.;	
    if(fabs(ref->eta()) < barrel_end_){
      etaBinLow = l1EGIso[i]->eta() - region_eta_size_/2.;
      etaBinHigh = etaBinLow + region_eta_size_;
    }
    else{
      etaBinLow = l1EGIso[i]->eta() - region_eta_size_ecap_/2.;
      etaBinHigh = etaBinLow + region_eta_size_ecap_;
    }

    float deltaphi=fabs(ref->phi() -l1EGIso[i]->phi());
    if(deltaphi>TWOPI) deltaphi-=TWOPI;
    if(deltaphi>M_PI) deltaphi=TWOPI-deltaphi;

    if(ref->eta() < etaBinHigh && ref->eta() > etaBinLow && deltaphi <region_phi_size_/2. )  {return true;}

  }

  for (unsigned int i=0; i<l1EGNonIso.size(); i++) {
    //ORCA matching method
    double etaBinLow  = 0.;
    double etaBinHigh = 0.;	
    if(fabs(ref->eta()) < barrel_end_){
      etaBinLow = l1EGNonIso[i]->eta() - region_eta_size_/2.;
      etaBinHigh = etaBinLow + region_eta_size_;
    }
    else{
      etaBinLow = l1EGNonIso[i]->eta() - region_eta_size_ecap_/2.;
      etaBinHigh = etaBinLow + region_eta_size_ecap_;
    }

    float deltaphi=fabs(ref->phi() - l1EGNonIso[i]->phi());
    if(deltaphi>TWOPI) deltaphi-=TWOPI;
    if(deltaphi>M_PI) deltaphi=TWOPI-deltaphi;

    if(ref->eta() < etaBinHigh && ref->eta() > etaBinLow && deltaphi <region_phi_size_/2. )  {return true;}
	
  }

  return false;
}
