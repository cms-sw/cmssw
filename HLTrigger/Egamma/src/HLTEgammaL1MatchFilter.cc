/** \class HLTEgammaL1MatchFilter
 *
 * $Id: HLTEgammaL1MatchFilter.cc,v 1.5 2007/03/23 16:52:09 ghezzi Exp $
 *
 *  \author Monica Vazquez Acosta (CERN)
 *
 */

#include "HLTrigger/Egamma/interface/HLTEgammaL1MatchFilter.h"

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
HLTEgammaL1MatchFilter::HLTEgammaL1MatchFilter(const edm::ParameterSet& iConfig)
{
   candTag_ = iConfig.getParameter< edm::InputTag > ("candTag");
   l1Tag_ = iConfig.getParameter< edm::InputTag > ("l1Tag");
   ncandcut_  = iConfig.getParameter<int> ("ncandcut");

   region_eta_size_      = iConfig.getParameter<double> ("region_eta_size");
   region_eta_size_ecap_ = iConfig.getParameter<double> ("region_eta_size_ecap");
   region_phi_size_      = iConfig.getParameter<double> ("region_phi_size");
   barrel_end_           = iConfig.getParameter<double> ("barrel_end");   
   endcap_end_           = iConfig.getParameter<double> ("endcap_end");   

   //register your products
   produces<reco::HLTFilterObjectWithRefs>();
}

HLTEgammaL1MatchFilter::~HLTEgammaL1MatchFilter(){}


// ------------ method called to produce the data  ------------
bool
HLTEgammaL1MatchFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  // The filter object
  std::auto_ptr<reco::HLTFilterObjectWithRefs> filterproduct (new reco::HLTFilterObjectWithRefs(path(),module()));
  // Ref to Candidate object to be recorded in filter object
  edm::RefToBase<reco::Candidate> ref;
  
  // Get the recoEcalCandidates
  edm::Handle<reco::RecoEcalCandidateCollection> recoecalcands;
  iEvent.getByLabel(candTag_,recoecalcands);
  //Get the L1 EM Particle Collection
  edm::Handle< l1extra::L1EmParticleCollection > emColl ;
  iEvent.getByLabel(l1Tag_, emColl ) ;
  // Get the CaloGeometry
  edm::ESHandle<L1CaloGeometry> l1CaloGeom ;
  iSetup.get<L1CaloGeometryRecord>().get(l1CaloGeom) ;


  // look at all candidates,  check cuts and add to filter object
  int n(0);

  for (reco::RecoEcalCandidateCollection::const_iterator recoecalcand= recoecalcands->begin(); recoecalcand!=recoecalcands->end(); recoecalcand++) {

    bool MATCHEDSC = false;

    if(fabs(recoecalcand->eta()) < endcap_end_){
      //SC should be inside the ECAL fiducial volume

      for( l1extra::L1EmParticleCollection::const_iterator emItr = emColl->begin(); emItr != emColl->end() ;++emItr ){


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
	ref=edm::RefToBase<reco::Candidate>(reco::RecoEcalCandidateRef(recoecalcands,distance(recoecalcands->begin(),recoecalcand)));
	filterproduct->putParticle(ref);
      }

    }
    
  }
  
  
  
  // filter decision
  bool accept(n>=ncandcut_);
  
  // put filter object into the Event
  iEvent.put(filterproduct);
  
  return accept;
}
