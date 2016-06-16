/** \class HLTEgammaL1TMatchFilterRegional
 *
 *
 *  \author Monica Vazquez Acosta (CERN)
 *
 */

#include "HLTrigger/Egamma/interface/HLTEgammaL1TMatchFilterRegional.h"

//#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

//#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
//#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/L1Trigger/interface/Jet.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/L1TObjects/interface/L1CaloGeometry.h"
#include "CondFormats/DataRecord/interface/L1CaloGeometryRecord.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"


#define TWOPI 6.283185308
//
// constructors and destructor
//
HLTEgammaL1TMatchFilterRegional::HLTEgammaL1TMatchFilterRegional(const edm::ParameterSet& iConfig) : HLTFilter(iConfig)
{
   candIsolatedTag_ = iConfig.getParameter< edm::InputTag > ("candIsolatedTag");
   l1EGTag_ = iConfig.getParameter< edm::InputTag > ("l1IsolatedTag"); //will be renamed l1EGTag for step 2 of the new L1 migration
   candNonIsolatedTag_ = iConfig.getParameter< edm::InputTag > ("candNonIsolatedTag");
   L1SeedFilterTag_ = iConfig.getParameter< edm::InputTag > ("L1SeedFilterTag"); 
   l1JetsTag_ = iConfig.getParameter< edm::InputTag > ("l1CenJetsTag"); //will be renamed l1JetsTag for step 2 of the new L1 migration
   l1TausTag_ = iConfig.getParameter< edm::InputTag > ("l1TausTag"); //will be renamed l1JetsTag for step 2 of the new L1 migration
   ncandcut_  = iConfig.getParameter<int> ("ncandcut");
   doIsolated_   = iConfig.getParameter<bool>("doIsolated");   
   region_eta_size_      = iConfig.getParameter<double> ("region_eta_size");
   region_eta_size_ecap_ = iConfig.getParameter<double> ("region_eta_size_ecap");
   region_phi_size_      = iConfig.getParameter<double> ("region_phi_size");
   barrel_end_           = iConfig.getParameter<double> ("barrel_end");
   endcap_end_           = iConfig.getParameter<double> ("endcap_end");

   candIsolatedToken_ = consumes<reco::RecoEcalCandidateCollection>(candIsolatedTag_);
   if(!doIsolated_) candNonIsolatedToken_ = consumes<reco::RecoEcalCandidateCollection>(candNonIsolatedTag_);
   L1SeedFilterToken_ = consumes<trigger::TriggerFilterObjectWithRefs>(L1SeedFilterTag_);
}

HLTEgammaL1TMatchFilterRegional::~HLTEgammaL1TMatchFilterRegional(){}

void
HLTEgammaL1TMatchFilterRegional::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("candIsolatedTag",edm::InputTag("hltRecoIsolatedEcalCandidate"));
  desc.add<edm::InputTag>("l1IsolatedTag",edm::InputTag("hltCaloStage2Digis")); //rename for step 2 of the L1 migration
  desc.add<edm::InputTag>("candNonIsolatedTag",edm::InputTag("hltRecoNonIsolatedEcalCandidate"));
  desc.add<edm::InputTag>("l1NonIsolatedTag",edm::InputTag("l1extraParticles","NonIsolated")); //drop for step 2 of the L1 migration
  desc.add<edm::InputTag>("L1SeedFilterTag",edm::InputTag("theL1SeedFilter"));
  desc.add<edm::InputTag>("l1CenJetsTag",edm::InputTag("hltCaloStage2Digis")); //rename for step 2 of L1 migration
  desc.add<edm::InputTag>("l1TausTag",edm::InputTag("hltCaloStage2Digis","Tau")); //rename for step 2 of L1 migration
  desc.add<int>("ncandcut",1);
  desc.add<bool>("doIsolated",true);
  desc.add<double>("region_eta_size",0.522);
  desc.add<double>("region_eta_size_ecap",1.0);
  desc.add<double>("region_phi_size",1.044);
  desc.add<double>("barrel_end",1.4791);
  desc.add<double>("endcap_end",2.65);
  descriptions.add("HLTEgammaL1TMatchFilterRegional",desc);
}

// ------------ method called to produce the data  ------------
//configuration:
//doIsolated=true, only isolated superclusters are allowed to match isolated L1 seeds
//doIsolated=false, isolated superclusters are allowed to match either iso or non iso L1 seeds, non isolated superclusters are allowed only to match non-iso seeds. If no collection name is given for non-isolated superclusters, assumes the the isolated collection contains all (both iso + non iso) seeded superclusters.
bool
HLTEgammaL1TMatchFilterRegional::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct) const
{
  // std::cout <<"runnr "<<iEvent.id().run()<<" event "<<iEvent.id().event()<<std::endl;
  using namespace trigger;
  //using namespace l1extra;

  if (saveTags()) {
    filterproduct.addCollectionTag(l1EGTag_);
    filterproduct.addCollectionTag(l1JetsTag_);
    filterproduct.addCollectionTag(l1TausTag_);
  }

  edm::Ref<reco::RecoEcalCandidateCollection> ref;

  // Get the CaloGeometry
  edm::ESHandle<L1CaloGeometry> l1CaloGeom ;
  iSetup.get<L1CaloGeometryRecord>().get(l1CaloGeom) ;


  // look at all candidates,  check cuts and add to filter object
  int n(0);

  // Get the recoEcalCandidates
  edm::Handle<reco::RecoEcalCandidateCollection> recoIsolecalcands;
  iEvent.getByToken(candIsolatedToken_,recoIsolecalcands);


  edm::Handle<trigger::TriggerFilterObjectWithRefs> L1SeedOutput;
  iEvent.getByToken (L1SeedFilterToken_,L1SeedOutput);

  std::vector<l1t::EGammaRef>  l1EGs;
  L1SeedOutput->getObjects(TriggerL1EG, l1EGs);

  std::vector<l1t::JetRef> l1Jets;
  L1SeedOutput->getObjects(TriggerL1Jet, l1Jets);

  std::vector<l1t::TauRef> l1Taus;
  L1SeedOutput->getObjects(TriggerL1Tau, l1Taus);

  int countCand=0;
  for (reco::RecoEcalCandidateCollection::const_iterator recoecalcand= recoIsolecalcands->begin(); recoecalcand!=recoIsolecalcands->end(); recoecalcand++) {
    countCand++;
    if(fabs(recoecalcand->eta()) < endcap_end_){
      //SC should be inside the ECAL fiducial volume

      //now EGamma is just one collection so automatically matches to Isolated and NonIsolated Seeds
      //it is assumed the HLTL1TSeed module fills it with the correct seeds
      bool matchedSCEG = matchedToL1Cand(l1EGs,recoecalcand->eta(),recoecalcand->phi());
      bool matchedSCJet = matchedToL1Cand(l1Jets,recoecalcand->eta(),recoecalcand->phi());
      bool matchedSCTau = matchedToL1Cand(l1Taus,recoecalcand->eta(),recoecalcand->phi());

      if(matchedSCEG || matchedSCJet || matchedSCTau) {
	n++;
	ref = edm::Ref<reco::RecoEcalCandidateCollection>(recoIsolecalcands, distance(recoIsolecalcands->begin(),recoecalcand) );
	filterproduct.addObject(TriggerCluster, ref);
      }//end  matched check

    }//end endcap fiduical check

  }//end loop over all isolated RecoEcalCandidates

  //if doIsolated_ is false now run over the nonisolated superclusters and EG
  //however in the case we have a single collection of superclusters containing both iso L1 and non iso L1 seeded superclusters,
  //we do not have a non isolated collection of superclusters so we have to protect against that
  if(!doIsolated_ && !candNonIsolatedTag_.label().empty()) {
    edm::Handle<reco::RecoEcalCandidateCollection> recoNonIsolecalcands;
    iEvent.getByToken(candNonIsolatedToken_,recoNonIsolecalcands);

    for (reco::RecoEcalCandidateCollection::const_iterator recoecalcand= recoNonIsolecalcands->begin(); recoecalcand!=recoNonIsolecalcands->end(); recoecalcand++) {
      countCand++;
 
      if(fabs(recoecalcand->eta()) < endcap_end_){
	bool matchedSCEG =  matchedToL1Cand(l1EGs,recoecalcand->eta(),recoecalcand->phi());
	bool matchedSCJet = matchedToL1Cand(l1Jets,recoecalcand->eta(),recoecalcand->phi());
	bool matchedSCTau = matchedToL1Cand(l1Taus,recoecalcand->eta(),recoecalcand->phi());
	
	if(matchedSCEG || matchedSCJet || matchedSCTau) {
	  n++;
	  ref = edm::Ref<reco::RecoEcalCandidateCollection>(recoNonIsolecalcands, distance(recoNonIsolecalcands->begin(),recoecalcand) );
	  filterproduct.addObject(TriggerCluster, ref);
	}//end  matched check
	
      }//end endcap fiduical check

    }//end loop over all isolated RecoEcalCandidates
  }//end doIsolatedCheck


  // filter decision
  bool accept(n>=ncandcut_);

  return accept;
}


bool
HLTEgammaL1TMatchFilterRegional::matchedToL1Cand(const std::vector<l1t::EGammaRef>& l1Cands,const float scEta,const float scPhi) const
{
  for (unsigned int i=0; i<l1Cands.size(); i++) {
    //ORCA matching method
    double etaBinLow  = 0.;
    double etaBinHigh = 0.;	
    if(fabs(scEta) < barrel_end_){
      etaBinLow = l1Cands[i]->eta() - region_eta_size_/2.;
      etaBinHigh = etaBinLow + region_eta_size_;
    }
    else{
      etaBinLow = l1Cands[i]->eta() - region_eta_size_ecap_/2.;
      etaBinHigh = etaBinLow + region_eta_size_ecap_;
    }

    float deltaphi=fabs(scPhi -l1Cands[i]->phi());
    if(deltaphi>TWOPI) deltaphi-=TWOPI;
    if(deltaphi>TWOPI/2.) deltaphi=TWOPI-deltaphi;

    if(scEta < etaBinHigh && scEta > etaBinLow && deltaphi <region_phi_size_/2. )  {
      return true;
    }
  }
  return false;
}

bool
//HLTEgammaL1TMatchFilterRegional::matchedToL1Cand(const std::vector<l1extra::L1JetParticleRef >& l1Cands,const float scEta,const float scPhi) const
HLTEgammaL1TMatchFilterRegional::matchedToL1Cand(const std::vector<l1t::JetRef>& l1Cands,const float scEta,const float scPhi) const
{
  for (unsigned int i=0; i<l1Cands.size(); i++) {
    //ORCA matching method
    double etaBinLow  = 0.;
    double etaBinHigh = 0.;	
    if(fabs(scEta) < barrel_end_){
      etaBinLow = l1Cands[i]->eta() - region_eta_size_/2.;
      etaBinHigh = etaBinLow + region_eta_size_;
    }
    else{
      etaBinLow = l1Cands[i]->eta() - region_eta_size_ecap_/2.;
      etaBinHigh = etaBinLow + region_eta_size_ecap_;
    }

    float deltaphi=fabs(scPhi -l1Cands[i]->phi());
    if(deltaphi>TWOPI) deltaphi-=TWOPI;
    if(deltaphi>TWOPI/2.) deltaphi=TWOPI-deltaphi;

    if(scEta < etaBinHigh && scEta > etaBinLow && deltaphi <region_phi_size_/2. )  {
      return true;
    }
  }
  return false;
}
bool
//HLTEgammaL1TMatchFilterRegional::matchedToL1Cand(const std::vector<l1extra::L1JetParticleRef >& l1Cands,const float scEta,const float scPhi) const
HLTEgammaL1TMatchFilterRegional::matchedToL1Cand(const std::vector<l1t::TauRef>& l1Cands,const float scEta,const float scPhi) const
{
  for (unsigned int i=0; i<l1Cands.size(); i++) {
    //ORCA matching method
    double etaBinLow  = 0.;
    double etaBinHigh = 0.;	
    if(fabs(scEta) < barrel_end_){
      etaBinLow = l1Cands[i]->eta() - region_eta_size_/2.;
      etaBinHigh = etaBinLow + region_eta_size_;
    }
    else{
      etaBinLow = l1Cands[i]->eta() - region_eta_size_ecap_/2.;
      etaBinHigh = etaBinLow + region_eta_size_ecap_;
    }

    float deltaphi=fabs(scPhi -l1Cands[i]->phi());
    if(deltaphi>TWOPI) deltaphi-=TWOPI;
    if(deltaphi>TWOPI/2.) deltaphi=TWOPI-deltaphi;

    if(scEta < etaBinHigh && scEta > etaBinLow && deltaphi <region_phi_size_/2. )  {
      return true;
    }
  }
  return false;
}
