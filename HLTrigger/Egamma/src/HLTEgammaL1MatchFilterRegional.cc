/** \class HLTEgammaL1MatchFilterRegional
 *
 * $Id: HLTEgammaL1MatchFilterRegional.cc,v 1.7 2007/12/09 13:31:41 ghezzi Exp $
 *
 *  \author Monica Vazquez Acosta (CERN)
 *
 */

#include "HLTrigger/Egamma/interface/HLTEgammaL1MatchFilterRegional.h"

//#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"

#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"

#include "CondFormats/L1TObjects/interface/L1CaloGeometry.h"
#include "CondFormats/DataRecord/interface/L1CaloGeometryRecord.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#define TWOPI 6.283185308
//
// constructors and destructor
//
HLTEgammaL1MatchFilterRegional::HLTEgammaL1MatchFilterRegional(const edm::ParameterSet& iConfig)
{
   candIsolatedTag_ = iConfig.getParameter< edm::InputTag > ("candIsolatedTag");
   l1IsolatedTag_ = iConfig.getParameter< edm::InputTag > ("l1IsolatedTag");
   candNonIsolatedTag_ = iConfig.getParameter< edm::InputTag > ("candNonIsolatedTag");
   l1NonIsolatedTag_ = iConfig.getParameter< edm::InputTag > ("l1NonIsolatedTag");
   L1SeedFilterTag_ = iConfig.getParameter< edm::InputTag > ("L1SeedFilterTag");
   ncandcut_  = iConfig.getParameter<int> ("ncandcut");
   doIsolated_   = iConfig.getParameter<bool>("doIsolated");


   region_eta_size_      = iConfig.getParameter<double> ("region_eta_size");
   region_eta_size_ecap_ = iConfig.getParameter<double> ("region_eta_size_ecap");
   region_phi_size_      = iConfig.getParameter<double> ("region_phi_size");
   barrel_end_           = iConfig.getParameter<double> ("barrel_end");   
   endcap_end_           = iConfig.getParameter<double> ("endcap_end");   

   //register your products
   produces<trigger::TriggerFilterObjectWithRefs>();
}

HLTEgammaL1MatchFilterRegional::~HLTEgammaL1MatchFilterRegional(){}


// ------------ method called to produce the data  ------------
bool
HLTEgammaL1MatchFilterRegional::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  using namespace trigger;
  using namespace l1extra;
  //using namespace std;
  std::auto_ptr<trigger::TriggerFilterObjectWithRefs> filterobject (new trigger::TriggerFilterObjectWithRefs(path(),module()));

  // std::auto_ptr<reco::HLTFilterObjectWithRefs> filterproduct (new reco::HLTFilterObjectWithRefs(path(),module()));
  // Ref to Candidate object to be recorded in filter object
  //edm::RefToBase<reco::Candidate> ref;
  //   edm::Ref<reco::RecoEcalCandidate> ref;// it does not work
   edm::Ref<reco::RecoEcalCandidateCollection> ref;

  // Get the CaloGeometry
  edm::ESHandle<L1CaloGeometry> l1CaloGeom ;
  iSetup.get<L1CaloGeometryRecord>().get(l1CaloGeom) ;


  // look at all candidates,  check cuts and add to filter object
  int n(0);

  // Get the recoEcalCandidates
  edm::Handle<reco::RecoEcalCandidateCollection> recoIsolecalcands;
  iEvent.getByLabel(candIsolatedTag_,recoIsolecalcands);

  //Get the L1 EM Particle Collection
  //edm::Handle< l1extra::L1EmParticleCollection > emIsolColl ;
  //iEvent.getByLabel(l1IsolatedTag_, emIsolColl ) ;

  edm::Handle<trigger::TriggerFilterObjectWithRefs> L1SeedOutput;

  iEvent.getByLabel (L1SeedFilterTag_,L1SeedOutput);

  std::vector<l1extra::L1EmParticleRef > l1EGIso;       
  L1SeedOutput->getObjects(TriggerL1IsoEG, l1EGIso);
  // std::cout<<"L1EGIso size: "<<l1EGIso.size()<<std::endl;

  for (reco::RecoEcalCandidateCollection::const_iterator recoecalcand= recoIsolecalcands->begin(); recoecalcand!=recoIsolecalcands->end(); recoecalcand++) {

    bool MATCHEDSC = false;

    if(fabs(recoecalcand->eta()) < endcap_end_){
      //SC should be inside the ECAL fiducial volume


      for (unsigned int i=0; i<l1EGIso.size(); i++) {
	//ORCA matching method
	double etaBinLow  = 0.;
	double etaBinHigh = 0.;	
	if(fabs(recoecalcand->eta()) < barrel_end_){
	  etaBinLow = l1EGIso[i]->eta() - region_eta_size_/2.;
	  etaBinHigh = etaBinLow + region_eta_size_;
	}
	else{
	  etaBinLow = l1EGIso[i]->eta() - region_eta_size_ecap_/2.;
	  etaBinHigh = etaBinLow + region_eta_size_ecap_;
	}

	float deltaphi=fabs(recoecalcand->phi() -l1EGIso[i]->phi());
	if(deltaphi>TWOPI) deltaphi-=TWOPI;
	if(deltaphi>TWOPI/2.) deltaphi=TWOPI-deltaphi;

	if(recoecalcand->eta() < etaBinHigh && recoecalcand->eta() > etaBinLow &&
	  deltaphi <region_phi_size_/2. )  {
	  MATCHEDSC = true;
	}
	
      }
      
      if(MATCHEDSC) {
	n++;
	//ref=edm::RefToBase<reco::Candidate>(reco::RecoEcalCandidateRef(recoIsolecalcands,distance(recoIsolecalcands->begin(),recoecalcand)));
	//filterproduct->putParticle(ref);
	ref = edm::Ref<reco::RecoEcalCandidateCollection>(recoIsolecalcands, distance(recoIsolecalcands->begin(),recoecalcand) );       
	filterobject->addObject(TriggerCluster, ref);
      }

    }
    
  }
  

  if(!doIsolated_) {
  edm::Handle<reco::RecoEcalCandidateCollection> recoNonIsolecalcands;
  iEvent.getByLabel(candNonIsolatedTag_,recoNonIsolecalcands);
  //Get the L1 EM Particle Collection
  //edm::Handle< l1extra::L1EmParticleCollection > emNonIsolColl ;
  //iEvent.getByLabel(l1NonIsolatedTag_, emNonIsolColl ) ;

  std::vector<l1extra::L1EmParticleRef > l1EGNonIso;       
  L1SeedOutput->getObjects(TriggerL1NoIsoEG, l1EGNonIso);
  //std::cout<<"L1EGNonIso size: "<<l1EGNonIso.size()<<std::endl;
  for (reco::RecoEcalCandidateCollection::const_iterator recoecalcand= recoNonIsolecalcands->begin(); recoecalcand!=recoNonIsolecalcands->end(); recoecalcand++) {

    bool MATCHEDSC = false;

    if(fabs(recoecalcand->eta()) < endcap_end_){
      //SC should be inside the ECAL fiducial volume

      for (unsigned int i=0; i<l1EGNonIso.size(); i++) {
	//ORCA matching method
	double etaBinLow  = 0.;
	double etaBinHigh = 0.;	
	if(fabs(recoecalcand->eta()) < barrel_end_){
	  etaBinLow = l1EGNonIso[i]->eta() - region_eta_size_/2.;
	  etaBinHigh = etaBinLow + region_eta_size_;
	}
	else{
	  etaBinLow = l1EGNonIso[i]->eta() - region_eta_size_ecap_/2.;
	  etaBinHigh = etaBinLow + region_eta_size_ecap_;
	}

	float deltaphi=fabs(recoecalcand->phi() - l1EGNonIso[i]->phi());
	if(deltaphi>TWOPI) deltaphi-=TWOPI;
	if(deltaphi>TWOPI/2.) deltaphi=TWOPI-deltaphi;

	if(recoecalcand->eta() < etaBinHigh && recoecalcand->eta() > etaBinLow &&
	  deltaphi <region_phi_size_/2. )  {
	  MATCHEDSC = true;
	}
	
      }
      
      if(MATCHEDSC) {
	n++;
	//ref=edm::RefToBase<reco::Candidate>(reco::RecoEcalCandidateRef(recoNonIsolecalcands,distance(recoNonIsolecalcands->begin(),recoecalcand)));
	//filterproduct->putParticle(ref);
	ref = edm::Ref<reco::RecoEcalCandidateCollection>(recoNonIsolecalcands, distance(recoNonIsolecalcands->begin(),recoecalcand) );       
	filterobject->addObject(TriggerCluster, ref);
      }

    }
    
  }
  }

  
  // filter decision
  bool accept(n>=ncandcut_);
  
  // put filter object into the Event
  iEvent.put(filterobject);
  
  return accept;
}
