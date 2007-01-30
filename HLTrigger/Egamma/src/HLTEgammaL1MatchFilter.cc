/** \class HLTEgammaL1MatchFilter
 *
 * $Id: $
 *
 *  \author Monica Vazquez Acosta (CERN)
 *
 */

#include "HLTrigger/Egamma/interface/HLTEgammaL1MatchFilter.h"

#include "FWCore/Framework/interface/Handle.h"

#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/HLTReco/interface/HLTFilterObject.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"

#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"

#include "L1TriggerConfig/L1Geometry/interface/L1CaloGeometry.h"
#include "L1TriggerConfig/L1Geometry/interface/L1CaloGeometryRecord.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"


//
// constructors and destructor
//
HLTEgammaL1MatchFilter::HLTEgammaL1MatchFilter(const edm::ParameterSet& iConfig)
{
   candTag_ = iConfig.getParameter< edm::InputTag > ("candTag");
   l1Tag_ = iConfig.getParameter< edm::InputTag > ("l1Tag");
   ncandcut_  = iConfig.getParameter<int> ("ncandcut");

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
  //const L1CaloGeometry* l1caloGeom = caloGeom.product();


  // look at all candidates,  check cuts and add to filter object
  int n(0);

  for( l1extra::L1EmParticleCollection::const_iterator emItr = emColl->begin(); emItr != emColl->end() ;++emItr ){
 
    // Access the GCT hardware object corresponding to the L1Extra EM object.
    int etaIndex = emItr->gctEmCand()->etaIndex() ;
    int phiIndex = emItr->gctEmCand()->phiIndex() ;

    // Use the L1CaloGeometry to find the eta, phi bin boundaries.
    double etaBinLow  = l1CaloGeom->etaBinLowEdge( etaIndex ) ;
    double etaBinHigh = l1CaloGeom->etaBinHighEdge( etaIndex ) ;
    double phiBinLow  = l1CaloGeom->emJetPhiBinLowEdge( phiIndex ) ;
    double phiBinHigh = l1CaloGeom->emJetPhiBinHighEdge( phiIndex ) ;

    for (reco::RecoEcalCandidateCollection::const_iterator recoecalcand= recoecalcands->begin(); recoecalcand!=recoecalcands->end(); recoecalcand++) {
	if(recoecalcand->eta() < etaBinHigh && recoecalcand->eta() > etaBinLow &&
	   recoecalcand->phi() < phiBinHigh && recoecalcand->phi() > phiBinLow){
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
