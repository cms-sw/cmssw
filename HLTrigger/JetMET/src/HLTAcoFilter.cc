/** \ class HLTAcoFilter
 *
 *
 *  \author Dominique J. Mangeol
 *
* acoplanar dphi(jet1,jet2),dphi(jet2,met),dphi(jet1,met)

*/

#include <string>
#include "HLTrigger/JetMET/interface/HLTAcoFilter.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/METReco/interface/CaloMET.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"


//
// constructors and destructor
//
HLTAcoFilter::HLTAcoFilter(const edm::ParameterSet& iConfig)
{
   inputJetTag_ = iConfig.getParameter< edm::InputTag > ("inputJetTag");
   inputMETTag_ = iConfig.getParameter< edm::InputTag > ("inputMETTag");
   saveTags_    = iConfig.getParameter<bool>("saveTags");
   minDPhi_     = iConfig.getParameter<double> ("minDeltaPhi");
   maxDPhi_     = iConfig.getParameter<double> ("maxDeltaPhi");
   minEtjet1_   = iConfig.getParameter<double> ("minEtJet1"); 
   minEtjet2_   = iConfig.getParameter<double> ("minEtJet2"); 
   AcoString_   = iConfig.getParameter<std::string> ("Acoplanar");

   //register your products
   produces<trigger::TriggerFilterObjectWithRefs>();
}

HLTAcoFilter::~HLTAcoFilter(){}


// ------------ method called to produce the data  ------------
bool
HLTAcoFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace std;
  using namespace edm;
  using namespace reco;
  using namespace trigger;

  // The filter object
  auto_ptr<trigger::TriggerFilterObjectWithRefs> 
    filterobject (new trigger::TriggerFilterObjectWithRefs(path(),module()));
  if (saveTags_) {
    filterobject->addCollectionTag(inputJetTag_);
    filterobject->addCollectionTag(inputMETTag_);
  }

  Handle<CaloJetCollection> recocalojets;
  iEvent.getByLabel(inputJetTag_,recocalojets);
  Handle<trigger::TriggerFilterObjectWithRefs> metcal;
  iEvent.getByLabel(inputMETTag_,metcal);

  // look at all candidates,  check cuts and add to filter object
  int n(0);
  int JetNum = recocalojets->size();

  // events with two or more jets

  double etjet1=0.;
  double etjet2=0.;
  double phijet1=0.;
  double phijet2=0.;
  double etmiss=0.;
  double phimiss=0.;
   
  VRcalomet vrefMET; 
  metcal->getObjects(TriggerMET,vrefMET);
  CaloMETRef metRef=vrefMET.at(0);
  etmiss  = vrefMET.at(0)->et();
  phimiss = vrefMET.at(0)->phi();

  CaloJetRef ref1,ref2;

  if (JetNum>0) {
    CaloJetCollection::const_iterator recocalojet = recocalojets->begin(); 
	
    etjet1 = recocalojet->et();
    phijet1 = recocalojet->phi();
    ref1  = CaloJetRef(recocalojets,distance(recocalojets->begin(),recocalojet));
    
    if(JetNum>1) {
      recocalojet++;
      etjet2 = recocalojet->et();
      phijet2 = recocalojet->phi();
      ref2  = CaloJetRef(recocalojets,distance(recocalojets->begin(),recocalojet));
    }
    double Dphi= -1.;
    int JetSel = 0;
    
    if (AcoString_ == "Jet2Met") {
      Dphi = fabs(phimiss-phijet2);
      if (JetNum>=2 && etjet1>minEtjet1_  && etjet2>minEtjet2_) {JetSel=1;} 
    }
    if (AcoString_ == "Jet1Jet2") {
      Dphi = fabs(phijet1-phijet2);
      if (JetNum>=2 && etjet1>minEtjet1_  && etjet2>minEtjet2_) {JetSel=1;} 
    }
    if (AcoString_ == "Jet1Met") { 
      Dphi = fabs(phimiss-phijet1);
      if (JetNum>=1 && etjet1>minEtjet1_ ) {JetSel=1;} 
    }
    
      
    if (Dphi>M_PI) {Dphi=2.0*M_PI-Dphi;}
    if(JetSel>0 && Dphi>=minDPhi_ && Dphi<=maxDPhi_){
      
      if (AcoString_=="Jet2Met" || AcoString_=="Jet1Met")  {filterobject->addObject(TriggerMET,metRef);}
      if (AcoString_=="Jet1Met" || AcoString_=="Jet1Jet2") {filterobject->addObject(TriggerJet,ref1);}
      if (AcoString_=="Jet2Met" || AcoString_=="Jet1Jet2") {filterobject->addObject(TriggerJet,ref2);}
      n++;
    }
    

  } // at least one jet
  
    
  // filter decision
  bool accept(n>=1);
    
  // put filter object into the Event
  iEvent.put(filterobject);
    
  return accept;
}
