/** \class HLTEgammaHcalIsolFilter
 *
 * $Id: HLTEgammaHcalIsolFilter.cc,v 1.10 2008/04/24 12:53:42 ghezzi Exp $
 *
 *  \author Monica Vazquez Acosta (CERN)
 *
 */

#include "HLTrigger/Egamma/interface/HLTEgammaHcalIsolFilter.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"


#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateIsolation.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"


//
// constructors and destructor
//
HLTEgammaHcalIsolFilter::HLTEgammaHcalIsolFilter(const edm::ParameterSet& iConfig)
{
  candTag_ = iConfig.getParameter< edm::InputTag > ("candTag");
  isoTag_ = iConfig.getParameter< edm::InputTag > ("isoTag");
  nonIsoTag_ = iConfig.getParameter< edm::InputTag > ("nonIsoTag");
  hcalisolbarrelcut_  = iConfig.getParameter<double> ("hcalisolbarrelcut");
  hcalisolendcapcut_  = iConfig.getParameter<double> ("hcalisolendcapcut");
  HoverEcut_          = iConfig.getParameter<double> ("HoverEcut");
  HoverEt2cut_          = iConfig.getParameter<double> ("HoverEt2cut");
  ncandcut_  = iConfig.getParameter<int> ("ncandcut");
  doIsolated_ = iConfig.getParameter<bool> ("doIsolated");

  store_ = iConfig.getUntrackedParameter<bool> ("SaveTag",false) ;
  L1IsoCollTag_= iConfig.getParameter< edm::InputTag > ("L1IsoCand"); 
  L1NonIsoCollTag_= iConfig.getParameter< edm::InputTag > ("L1NonIsoCand"); 

  //register your products
  produces<trigger::TriggerFilterObjectWithRefs>();
}

HLTEgammaHcalIsolFilter::~HLTEgammaHcalIsolFilter(){}

// ------------ method called to produce the data  ------------
bool
HLTEgammaHcalIsolFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   // The filter object
  using namespace trigger;
    std::auto_ptr<trigger::TriggerFilterObjectWithRefs> filterproduct (new trigger::TriggerFilterObjectWithRefs(path(),module()));
    if( store_ ){filterproduct->addCollectionTag(L1IsoCollTag_);}
    if( store_ && !doIsolated_){filterproduct->addCollectionTag(L1NonIsoCollTag_);}
  // Ref to Candidate object to be recorded in filter object
   edm::Ref<reco::RecoEcalCandidateCollection> ref;


  edm::Handle<trigger::TriggerFilterObjectWithRefs> PrevFilterOutput;

  iEvent.getByLabel (candTag_,PrevFilterOutput);

  std::vector<edm::Ref<reco::RecoEcalCandidateCollection> > recoecalcands;       
  PrevFilterOutput->getObjects(TriggerCluster, recoecalcands);

  
  //get hold of hcal isolation association map
  edm::Handle<reco::RecoEcalCandidateIsolationMap> depMap;
  iEvent.getByLabel (isoTag_,depMap);
  
  //get hold of hcal isolation association map
  edm::Handle<reco::RecoEcalCandidateIsolationMap> depNonIsoMap;
  if(!doIsolated_) iEvent.getByLabel (nonIsoTag_,depNonIsoMap);

  // look at all photons,  check cuts and add to filter object
  int n = 0;
  
  for (unsigned int i=0; i<recoecalcands.size(); i++) {

     ref = recoecalcands[i] ;
    //std::cout<<"MARCO HLTEgammaHcalIsolFilter i "<<i<<" "<<std::endl;
    //std::cout<<"MARCO HLTEgammaHcalIsolFilter candref "<<(long) ref<<" "<<std::endl;    
     //  reco::RecoEcalCandidateRef recr = ref.castTo<reco::RecoEcalCandidateRef>();
    //std::cout<<"MARCO HLTEgammaHcalIsolFilter recr "<<recr<<" "<<std::endl;
    
    reco::RecoEcalCandidateIsolationMap::const_iterator mapi = (*depMap).find( ref );
    
    if(mapi==(*depMap).end()) {
      if(!doIsolated_) mapi = (*depNonIsoMap).find( ref ); 
      //std::cout<<"MARCO HLTEgammaEcalIsolFilter 100 "<<std::endl;
    }
     float vali = mapi->val;
     //std::cout<<"MARCO HLTEgammaHcalIsolFilter vali "<<vali<<" "<<std::endl;
     float HoE = mapi->val / ref->et();
     float HoE2 = HoE / ref->et();

     if(fabs(ref->eta()) < 1.5){
       if ( vali < hcalisolbarrelcut_ || HoE < HoverEcut_ || HoE2 < HoverEt2cut_ ) {
	 n++;
	 filterproduct->addObject(TriggerCluster, ref);
       }
     }
     if(
	( fabs(ref->eta()) >= 1.5) && 
	( fabs(ref->eta()) < 2.5)
	){
       if ( vali < hcalisolendcapcut_ || HoE < HoverEcut_ || HoE2 < HoverEt2cut_ ) {
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

