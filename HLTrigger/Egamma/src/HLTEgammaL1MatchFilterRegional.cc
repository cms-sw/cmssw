/** \class HLTEgammaL1MatchFilterRegional
 *
 * $Id: HLTEgammaL1MatchFilterRegional.cc,v 1.1 2007/04/02 17:14:14 mpieri Exp $
 *
 *  \author Monica Vazquez Acosta (CERN)
 *
 */

#include "HLTrigger/Egamma/interface/HLTEgammaL1MatchFilterRegional.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/HLTReco/interface/HLTFilterObject.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"

#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"

#include "L1TriggerConfig/L1Geometry/interface/L1CaloGeometry.h"
#include "L1TriggerConfig/L1Geometry/interface/L1CaloGeometryRecord.h"

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
   ncandcut_  = iConfig.getParameter<int> ("ncandcut");
   doIsolated_   = iConfig.getParameter<bool>("doIsolated");


   region_eta_size_      = iConfig.getParameter<double> ("region_eta_size");
   region_eta_size_ecap_ = iConfig.getParameter<double> ("region_eta_size_ecap");
   region_phi_size_      = iConfig.getParameter<double> ("region_phi_size");
   barrel_end_           = iConfig.getParameter<double> ("barrel_end");   
   endcap_end_           = iConfig.getParameter<double> ("endcap_end");   

   //register your products
   produces<reco::HLTFilterObjectWithRefs>();
}

HLTEgammaL1MatchFilterRegional::~HLTEgammaL1MatchFilterRegional(){}


// ------------ method called to produce the data  ------------
bool
HLTEgammaL1MatchFilterRegional::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  // The filter object
  std::auto_ptr<reco::HLTFilterObjectWithRefs> filterproduct (new reco::HLTFilterObjectWithRefs(path(),module()));
  // Ref to Candidate object to be recorded in filter object
  edm::RefToBase<reco::Candidate> ref;
  

  // Get the CaloGeometry
  edm::ESHandle<L1CaloGeometry> l1CaloGeom ;
  iSetup.get<L1CaloGeometryRecord>().get(l1CaloGeom) ;


  // look at all candidates,  check cuts and add to filter object
  int n(0);

  // Get the recoEcalCandidates
  edm::Handle<reco::RecoEcalCandidateCollection> recoIsolecalcands;
  iEvent.getByLabel(candIsolatedTag_,recoIsolecalcands);
  //Get the L1 EM Particle Collection
  edm::Handle< l1extra::L1EmParticleCollection > emIsolColl ;
  iEvent.getByLabel(l1IsolatedTag_, emIsolColl ) ;

  for (reco::RecoEcalCandidateCollection::const_iterator recoecalcand= recoIsolecalcands->begin(); recoecalcand!=recoIsolecalcands->end(); recoecalcand++) {

    bool MATCHEDSC = false;

    if(fabs(recoecalcand->eta()) < endcap_end_){
      //SC should be inside the ECAL fiducial volume

      for( l1extra::L1EmParticleCollection::const_iterator emItr = emIsolColl->begin(); emItr != emIsolColl->end() ;++emItr ){


	//ORCA matching method
	double etaBinLow  = 0.;
	double etaBinHigh = 0.;	
	if(fabs(recoecalcand->eta()) < barrel_end_){
	  etaBinLow = emItr->eta() - region_eta_size_/2.;
	  etaBinHigh = etaBinLow + region_eta_size_;
	}
	else{
	  etaBinLow = emItr->eta() - region_eta_size_ecap_/2.;
	  etaBinHigh = etaBinLow + region_eta_size_ecap_;
	}

	float deltaphi=fabs(recoecalcand->phi() - emItr->phi());
	if(deltaphi>TWOPI) deltaphi-=TWOPI;
	if(deltaphi>TWOPI/2.) deltaphi=TWOPI-deltaphi;

	if(recoecalcand->eta() < etaBinHigh && recoecalcand->eta() > etaBinLow &&
	  deltaphi <region_phi_size_/2. )  {
	  MATCHEDSC = true;
	}
	
      }
      
      if(MATCHEDSC) {
	n++;
	ref=edm::RefToBase<reco::Candidate>(reco::RecoEcalCandidateRef(recoIsolecalcands,distance(recoIsolecalcands->begin(),recoecalcand)));
	filterproduct->putParticle(ref);
      }

    }
    
  }
  

  if(!doIsolated_) {
  edm::Handle<reco::RecoEcalCandidateCollection> recoNonIsolecalcands;
  iEvent.getByLabel(candNonIsolatedTag_,recoNonIsolecalcands);
  //Get the L1 EM Particle Collection
  edm::Handle< l1extra::L1EmParticleCollection > emNonIsolColl ;
  iEvent.getByLabel(l1NonIsolatedTag_, emNonIsolColl ) ;

  for (reco::RecoEcalCandidateCollection::const_iterator recoecalcand= recoNonIsolecalcands->begin(); recoecalcand!=recoNonIsolecalcands->end(); recoecalcand++) {

    bool MATCHEDSC = false;

    if(fabs(recoecalcand->eta()) < endcap_end_){
      //SC should be inside the ECAL fiducial volume

      for( l1extra::L1EmParticleCollection::const_iterator emItr = emNonIsolColl->begin(); emItr != emNonIsolColl->end() ;++emItr ){


	//ORCA matching method
	double etaBinLow  = 0.;
	double etaBinHigh = 0.;	
	if(fabs(recoecalcand->eta()) < barrel_end_){
	  etaBinLow = emItr->eta() - region_eta_size_/2.;
	  etaBinHigh = etaBinLow + region_eta_size_;
	}
	else{
	  etaBinLow = emItr->eta() - region_eta_size_ecap_/2.;
	  etaBinHigh = etaBinLow + region_eta_size_ecap_;
	}

	float deltaphi=fabs(recoecalcand->phi() - emItr->phi());
	if(deltaphi>TWOPI) deltaphi-=TWOPI;
	if(deltaphi>TWOPI/2.) deltaphi=TWOPI-deltaphi;

	if(recoecalcand->eta() < etaBinHigh && recoecalcand->eta() > etaBinLow &&
	  deltaphi <region_phi_size_/2. )  {
	  MATCHEDSC = true;
	}
	
      }
      
      if(MATCHEDSC) {
	n++;
	ref=edm::RefToBase<reco::Candidate>(reco::RecoEcalCandidateRef(recoNonIsolecalcands,distance(recoNonIsolecalcands->begin(),recoecalcand)));
	filterproduct->putParticle(ref);
      }

    }
    
  }
  }



  
  
  // filter decision
  bool accept(n>=ncandcut_);
  
  // put filter object into the Event
  iEvent.put(filterproduct);
  
  return accept;
}
