/** \class HLTEgammaClusterShapeFilter
 *
 * $Id: HLTEgammaClusterShapeFilter.cc,v 1.2 2009/01/15 14:31:49 covarell Exp $
 *
 *  \author Monica Vazquez Acosta (CERN)
 *
 */

#include "HLTrigger/Egamma/interface/HLTEgammaClusterShapeFilter.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"


#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateIsolation.h"

//////////////////////////////////////////////////////
//
// constructors and destructor
//
HLTEgammaClusterShapeFilter::HLTEgammaClusterShapeFilter(const edm::ParameterSet& iConfig)
{
  candTag_ = iConfig.getParameter< edm::InputTag > ("candTag");
  isoTag_ = iConfig.getParameter< edm::InputTag > ("isoTag");
  nonIsoTag_ = iConfig.getParameter< edm::InputTag > ("nonIsoTag");

  thresholdEB_ = iConfig.getParameter<double> ("BarrelThreshold");
  thresholdEE_ = iConfig.getParameter<double> ("EndcapThreshold");
  ncandcut_  = iConfig.getParameter<int> ("ncandcut");
  doIsolated_ = iConfig.getParameter<bool> ("doIsolated");

  store_ = iConfig.getUntrackedParameter<bool> ("SaveTag",false) ;
  L1IsoCollTag_= iConfig.getParameter< edm::InputTag > ("L1IsoCand"); 
  L1NonIsoCollTag_= iConfig.getParameter< edm::InputTag > ("L1NonIsoCand"); 

  //register your products
  produces<trigger::TriggerFilterObjectWithRefs>();
}

HLTEgammaClusterShapeFilter::~HLTEgammaClusterShapeFilter(){}

// ------------ method called to produce the data  ------------
bool
HLTEgammaClusterShapeFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   // The filter object
  using namespace trigger;
  using namespace edm;
  std::auto_ptr<trigger::TriggerFilterObjectWithRefs> filterproduct (new trigger::TriggerFilterObjectWithRefs(path(),module()));
  if( store_ ){filterproduct->addCollectionTag(L1IsoCollTag_);}
  if( store_ && !doIsolated_){filterproduct->addCollectionTag(L1NonIsoCollTag_);}
  
  // Ref to Candidate object to be recorded in filter object
  edm::Ref<reco::RecoEcalCandidateCollection> ref;

  edm::Handle<trigger::TriggerFilterObjectWithRefs> PrevFilterOutput;
  iEvent.getByLabel (candTag_,PrevFilterOutput);

  std::vector<edm::Ref<reco::RecoEcalCandidateCollection> > recoecalcands;       
  PrevFilterOutput->getObjects(TriggerCluster, recoecalcands);

  // retrieve cluster shape association map (iso)
  edm::Handle<reco::RecoEcalCandidateIsolationMap> depMap;
  iEvent.getByLabel (isoTag_,depMap);
  
  // retrieve cluster shape association map (non iso)
  edm::Handle<reco::RecoEcalCandidateIsolationMap> depNonIsoMap;
  if (!doIsolated_) iEvent.getByLabel (nonIsoTag_,depNonIsoMap);

  // look at all SC,  check cuts and add to filter object
  int n = 0;
  
  for (unsigned int i=0; i<recoecalcands.size(); i++) {

    ref = recoecalcands[i] ;
     
    reco::RecoEcalCandidateIsolationMap::const_iterator mapi = (*depMap).find( ref );
    if( mapi == (*depMap).end() && !doIsolated_) mapi = (*depNonIsoMap).find( ref ); 

    float EtaSC = fabs(ref->eta());
    float sigmaee = mapi->val;

    if(EtaSC < 1.479 ) {  // Barrel
      if (sigmaee < thresholdEB_ ) {
	n++;
	filterproduct->addObject(TriggerCluster, ref); 
      } 
    } 
    else {  //Endcap
      // sigmaee = sigmaee - 0.02*(EtaSC - 2.3);  // correction moved to producer
      if (sigmaee < thresholdEE_ ) {
	n++;
	filterproduct->addObject(TriggerCluster, ref); 
      }      
    } 
     
  }//end of loop ofver recoecalcands 
  
   // filter decision
   bool accept(n>=ncandcut_);

   // put filter object into the Event
   iEvent.put(filterproduct);

   return accept;
}

