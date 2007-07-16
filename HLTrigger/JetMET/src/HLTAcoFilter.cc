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

#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/HLTReco/interface/HLTFilterObject.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/JetReco/interface/CaloJet.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"


//
// constructors and destructor
//
HLTAcoFilter::HLTAcoFilter(const edm::ParameterSet& iConfig)
{
   inputJetTag_ = iConfig.getParameter< edm::InputTag > ("inputJetTag");
   inputMETTag_ = iConfig.getParameter< edm::InputTag > ("inputMETTag");
   minDPhi_   = iConfig.getParameter<double> ("minDeltaPhi");
   maxDPhi_   = iConfig.getParameter<double> ("maxDeltaPhi");
   minEtjet1_= iConfig.getParameter<double> ("minEtJet1"); 
   minEtjet2_= iConfig.getParameter<double> ("minEtJet2"); 
   AcoString_ = iConfig.getParameter<std::string> ("Acoplanar");

   //register your products
   produces<reco::HLTFilterObjectWithRefs>();
}

HLTAcoFilter::~HLTAcoFilter(){}


// ------------ method called to produce the data  ------------
bool
HLTAcoFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
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
int JetNum = recocalojets->size();

    // events with two or more jets

    double etjet1=0.;
    double etjet2=0.;
    double phijet1=0.;
    double phijet2=0.;
    double etmiss=0.;
    double phimiss=0.;
    // int countjets =0;
   
    //ccla HLTParticle met;
    Particle met;
     met=metcal->getParticle(0);
     metref=metcal->getParticleRef(0);
     etmiss=met.et();
     phimiss = met.phi();
if (JetNum>0) {
     CaloJetCollection::const_iterator recocalojet = recocalojets->begin(); 
	
               	etjet1 = recocalojet->et();
                phijet1 = recocalojet->phi();
                ref1  = RefToBase<Candidate>(CaloJetRef(recocalojets,distance(recocalojets->begin(),recocalojet)));
      
      if(JetNum>1) {
                recocalojet++;
	etjet2 = recocalojet->et();
                phijet2 = recocalojet->phi();
                ref2  = RefToBase<Candidate>(CaloJetRef(recocalojets,distance(recocalojets->begin(),recocalojet)));
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


    if (Dphi>PI) {Dphi=2.0*PI-Dphi;}
    if(JetSel>0 && Dphi>=minDPhi_ && Dphi<=maxDPhi_){

	if (AcoString_=="Jet2Met" || AcoString_=="Jet1Met")  {filterproduct->putParticle(metref);}
	if (AcoString_=="Jet1Met" || AcoString_=="Jet1Jet2") {filterproduct->putParticle(ref1);}
	if (AcoString_=="Jet2Met" || AcoString_=="Jet1Jet2") {filterproduct->putParticle(ref2);}
	n++;
    }
    

  } /// at least one jet
  
  
  // filter decision
  bool accept(n>=1);
  
  // put filter object into the Event
  iEvent.put(filterproduct);
  
  return accept;
}
