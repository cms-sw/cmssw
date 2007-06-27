/** \class HLTPhi2METFilter
 *
 *
 *  \author Dominique J. Mangeol
 *
 */

#include "HLTrigger/JetMET/interface/HLTPhi2METFilter.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/HLTReco/interface/HLTFilterObject.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/JetReco/interface/CaloJet.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"


//
// constructors and destructor
//
HLTPhi2METFilter::HLTPhi2METFilter(const edm::ParameterSet& iConfig)
{
   inputJetTag_ = iConfig.getParameter< edm::InputTag > ("inputJetTag");
   inputMETTag_ = iConfig.getParameter< edm::InputTag > ("inputMETTag");
   minDPhi_   = iConfig.getParameter<double> ("minDeltaPhi");
   maxDPhi_   = iConfig.getParameter<double> ("maxDeltaPhi");
   minEtjet1_= iConfig.getParameter<double> ("minEtJet1"); 
   minEtjet2_= iConfig.getParameter<double> ("minEtJet2"); 

   //register your products
   produces<reco::HLTFilterObjectWithRefs>();
}

HLTPhi2METFilter::~HLTPhi2METFilter(){}


// ------------ method called to produce the data  ------------
bool
HLTPhi2METFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace std;
  using namespace edm;
  using namespace reco;
  double PI=3.1415926654;
  // The filter object
  auto_ptr<HLTFilterObjectWithRefs> filterproduct (new HLTFilterObjectWithRefs(path(),module()));
  // Ref to Candidate object to be recorded in filter object
  RefToBase<Candidate> ref1,ref2,metref;
  // Get the Candidates

  Handle<CaloJetCollection> recocalojets;
  iEvent.getByLabel(inputJetTag_,recocalojets);
  Handle<HLTFilterObjectWithRefs> metcal;
  iEvent.getByLabel(inputMETTag_,metcal);


  // look at all candidates,  check cuts and add to filter object
  int n(0);

  if(recocalojets->size() > 1){
    // events with two or more jets

    double etjet1=0.;
    double etjet2=0.;
    double phijet2=0.;
    double etmiss=0.;
    double phimiss=0.;
    int countjets =0;
   
    //ccla HLTParticle met;
    Particle met;
     met=metcal->getParticle(0);
     metref=metcal->getParticleRef(0);
     etmiss=met.et();
     phimiss = met.phi();
    for (CaloJetCollection::const_iterator recocalojet = recocalojets->begin(); 
	 recocalojet<=(recocalojets->begin()+1); recocalojet++) {
      
      if(countjets==0) {
	etjet1 = recocalojet->et();
                ref1  = RefToBase<Candidate>(CaloJetRef(recocalojets,distance(recocalojets->begin(),recocalojet)));
      }
      if(countjets==1) {
	etjet2 = recocalojet->et();
                phijet2 = recocalojet->phi();
                ref2  = RefToBase<Candidate>(CaloJetRef(recocalojets,distance(recocalojets->begin(),recocalojet)));
      }
      countjets++;
    }
    double Dphi= fabs(phimiss-phijet2);
    if (Dphi>PI) Dphi=2.0*PI-Dphi;
    if(etjet1>minEtjet1_  && etjet2>minEtjet2_ && Dphi>=minDPhi_ && Dphi<=maxDPhi_){
	filterproduct->putParticle(metref);
	filterproduct->putParticle(ref1);
	filterproduct->putParticle(ref2);
	n++;
    }
    
  } // events with two or more jets
  
  
  
  // filter decision
  bool accept(n>=1);
  
  // put filter object into the Event
  iEvent.put(filterproduct);
  
  return accept;
}
