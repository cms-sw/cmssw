/** \class HLTExclDiJetFilter
 *
 *
 *  \author Leonard Apanasevich
 *
 */

#include "HLTrigger/JetMET/interface/HLTExclDiJetFilter.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/JetReco/interface/CaloJetCollection.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

//#include "DataFormats/Math/interface/deltaPhi.h"
#include <cmath>

#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"

//
// constructors and destructor
//
HLTExclDiJetFilter::HLTExclDiJetFilter(const edm::ParameterSet& iConfig)
{
   inputJetTag_ = iConfig.getParameter< edm::InputTag > ("inputJetTag");
   saveTag_     = iConfig.getUntrackedParameter<bool>("saveTag");
   minPtJet_    = iConfig.getParameter<double> ("minPtJet"); 
   minHFe_      = iConfig.getParameter<double> ("minHFe"); 
   HF_OR_       = iConfig.getParameter<bool> ("HF_OR"); 
   //register your products
   produces<trigger::TriggerFilterObjectWithRefs>();
}

HLTExclDiJetFilter::~HLTExclDiJetFilter(){}

void
HLTExclDiJetFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("inputJetTag",edm::InputTag("hltMCJetCorJetIcone5HF07"));
  desc.addUntracked<bool>("saveTag",false);
  desc.add<double>("minPtJet",30.0);
  desc.add<double>("minHFe",50.0);
  desc.add<bool>("HF_OR",false);
  descriptions.add("hltExclDiJetFilter",desc);
}

// ------------ method called to produce the data  ------------
bool
HLTExclDiJetFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace std;
  using namespace edm;
  using namespace reco;
  using namespace trigger;


  // The filter object
  auto_ptr<trigger::TriggerFilterObjectWithRefs> 
    filterobject (new trigger::TriggerFilterObjectWithRefs(path(),module()));
  if (saveTag_) filterobject->addCollectionTag(inputJetTag_);

  Handle<CaloJetCollection> recocalojets;
  iEvent.getByLabel(inputJetTag_,recocalojets);

  // look at all candidates,  check cuts and add to filter object
    int n(0);

    double ptjet1=0., ptjet2=0.;  
    double phijet1=0., phijet2=0.;

  if(recocalojets->size() > 1){
    // events with two or more jets

    int countjets =0;

    CaloJetRef JetRef1,JetRef2;

    for (CaloJetCollection::const_iterator recocalojet = recocalojets->begin(); 
	 recocalojet<=(recocalojets->begin()+1); ++recocalojet) {
      
      if(countjets==0) {
	ptjet1 = recocalojet->pt();
        phijet1 = recocalojet->phi();

	JetRef1 = CaloJetRef(recocalojets,distance(recocalojets->begin(),recocalojet));
      }
      if(countjets==1) {
	ptjet2 = recocalojet->pt();
        phijet2 = recocalojet->phi(); 

	JetRef2 = CaloJetRef(recocalojets,distance(recocalojets->begin(),recocalojet));
      }

      ++countjets;
    }

    if(ptjet1>minPtJet_ && ptjet2>minPtJet_ ){
      double Dphi=fabs(phijet1-phijet2);
      if(Dphi>M_PI) Dphi=2.0*M_PI-Dphi;
    if(Dphi>0.5*M_PI) {
       filterobject->addObject(TriggerJet,JetRef1);
       filterobject->addObject(TriggerJet,JetRef2);
       ++n;
    }
    }

  } // events with two or more jets

// calotowers

  bool hf_accept=false; 

  if(n>0) {
     double ehfp(0.);
     double ehfm(0.);

     Handle<CaloTowerCollection> o;
     iEvent.getByLabel("hltTowerMakerForAll",o);
//     if( o.isValid()) {
      for( CaloTowerCollection::const_iterator cc = o->begin(); cc != o->end(); ++cc ) {
       if(fabs(cc->ieta())>28 && cc->energy()<4.0) continue;
        if(cc->ieta()>28)  ehfp+=cc->energy();  // HF+ energy
        if(cc->ieta()<-28) ehfm+=cc->energy();  // HF- energy
      }
 //    }

     bool hf_accept_and  = (ehfp<minHFe_) && (ehfm<minHFe_);
     bool hf_accept_or  = (ehfp<minHFe_) || (ehfm<minHFe_);

     hf_accept = HF_OR_ ? hf_accept_or : hf_accept_and;

  } // n>0


////////////////////////////////////////////////////////  
  
// filter decision
  bool accept(n>0 && hf_accept);

  // put filter object into the Event
  iEvent.put(filterobject);
  
  return accept;
}
