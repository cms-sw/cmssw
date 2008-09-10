/** \class HLTEgammaClusterShapeFilter
 *
 * $Id: HLTEgammaClusterShapeFilter.cc,v 1.11 2008/04/25 15:18:51 ghezzi Exp $
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

//////////////////////////////////////////////////////
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"
//
// constructors and destructor
//
HLTEgammaClusterShapeFilter::HLTEgammaClusterShapeFilter(const edm::ParameterSet& iConfig)
{
  candTag_ = iConfig.getParameter< edm::InputTag > ("candTag");
  
  ecalRechitEBTag_ = iConfig.getParameter< edm::InputTag > ("ecalRechitEB");
  ecalRechitEETag_ = iConfig.getParameter< edm::InputTag > ("ecalRechitEE");

  thresholdEB_ = iConfig.getParameter<double> ("BarrelThreshold");
  thresholdEE_ = iConfig.getParameter<double> ("EndcapThreshold");
  ncandcut_  = iConfig.getParameter<int> ("ncandcut");
  //doIsolated_ = iConfig.getParameter<bool> ("doIsolated");

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

  EcalClusterLazyTools lazyTools( iEvent, iSetup, ecalRechitEBTag_, ecalRechitEETag_ );
  // look at all SC,  check cuts and add to filter object
  int n = 0;
  
  for (unsigned int i=0; i<recoecalcands.size(); i++) {

     ref = recoecalcands[i] ;
     
    //std::cout<<"MARCO HLTEgammaClusterShapeFilter i "<<i<<" "<<std::endl;
    //std::cout<<"MARCO HLTEgammaClusterShapeFilter candref "<<(long) ref<<" "<<std::endl;    
     //  reco::RecoEcalCandidateRef recr = ref.castTo<reco::RecoEcalCandidateRef>();
    //std::cout<<"MARCO HLTEgammaClusterShapeFilter recr "<<recr<<" "<<std::endl;
    
       //for(reco::SuperClusterCollection::const_iterator SCit = scHandle->begin(); SCit != scHandle->end(); SCit++) {
    std::vector<float> vCov = lazyTools.covariances( *(ref->superCluster()->seed()) );
    
    double sigmaee = sqrt(vCov[0]);
    float EtaSC = fabs(ref->eta());
    if(EtaSC < 1.479 ) {//Barrel
      if (sigmaee < thresholdEB_ ) {
	n++;
	filterproduct->addObject(TriggerCluster, ref);
      }
    }
    else {//Endcap
      sigmaee = sigmaee - 0.02*(EtaSC - 2.3);
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

