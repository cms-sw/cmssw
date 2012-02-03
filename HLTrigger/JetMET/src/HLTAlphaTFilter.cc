/** \class HLTAlphaTFilter
*
*
*  \author Bryn Mathias
*
*/

#include "HLTrigger/JetMET/interface/HLTAlphaTFilter.h"
#include "HLTrigger/JetMET/interface/AlphaT.hh"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/JetReco/interface/CaloJetCollection.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Math/interface/deltaPhi.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include <vector>
#include <algorithm>
#include <functional>
#include <numeric>
#include "TLorentzVector.h"


typedef ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double> > LorentzV  ;

//
// constructors and destructor
//
HLTAlphaTFilter::HLTAlphaTFilter(const edm::ParameterSet& iConfig) :

  inputJetTag_            ( iConfig.getParameter< edm::InputTag > ("inputJetTag") ),
  inputJetTagFastJet_     ( iConfig.getParameter< edm::InputTag > ("inputJetTagFastJet") ),
  minPtJet_               ( iConfig.getParameter<std::vector<double> > ("minPtJet") ),
  etaJet_                 ( iConfig.getParameter<std::vector<double> > ("etaJet") ),
  minHt_                  ( iConfig.getParameter<double> ("minHt") ),
  minAlphaT_              ( iConfig.getParameter<double> ("minAlphaT") )
// sanity checks
  {
  if (       (minPtJet_.size()    !=  etaJet_.size())
  || (  (minPtJet_.size()<1) || (etaJet_.size()<1) )
  || ( ((minPtJet_.size()<2) || (etaJet_.size()<2))))
  {
  edm::LogError("HLTAlphaTFilter") << "inconsistent module configuration!";
  }

//register your products
  produces<trigger::TriggerFilterObjectWithRefs>();
}

HLTAlphaTFilter::~HLTAlphaTFilter(){}

void HLTAlphaTFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("inputJetTag",edm::InputTag("hltMCJetCorJetIcone5HF07"));
  desc.add<edm::InputTag>("inputJetTagFastJet",edm::InputTag("hltMCJetCorJetIcone5HF07"));

  {
    std::vector<double> temp1;
    temp1.reserve(2);
    temp1.push_back(20.0);
    temp1.push_back(20.0);
    desc.add<std::vector<double> >("minPtJet",temp1);
  }
  desc.add<int>("minNJet",0);
  {
    std::vector<double> temp1;
    temp1.reserve(2);
    temp1.push_back(9999.0);
    temp1.push_back(9999.0);
    desc.add<std::vector<double> >("etaJet",temp1);
  }

  desc.add<double>("minHt",0.0);
  desc.add<double>("minAlphaT",0.0);
  descriptions.add("hltAlphaTFilter",desc);
}



// ------------ method called to produce the data  ------------
bool HLTAlphaTFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
using namespace std;
using namespace edm;
using namespace reco;
using namespace trigger;
// The filter object
// auto_ptr<trigger::TriggerFilterObjectWithRefs> filterobject (new trigger::TriggerFilterObjectWithRefs(path(),module()));
  
  CaloJetRef ref;
  // Get the Candidates
  Handle<CaloJetCollection> recocalojets;
  iEvent.getByLabel(inputJetTag_,recocalojets);


  // We have to also look at the L1 FastJet Corrections, at the same time we look at our other jets.
  // We calcualte our HT from the FastJet collection and AlphaT from the standard collection.
  CaloJetRef ref_FastJet;
  // Get the Candidates
  Handle<CaloJetCollection> recocalojetsFastJet;
  iEvent.getByLabel(inputJetTagFastJet_,recocalojetsFastJet);





// look at all candidates,  check cuts and add to filter object
  int n(0), flag(0);
  double htFast = 0.;

if(recocalojets->size() > 1){
  // events with at least two jets, needed for alphaT
  // Make a vector of Lorentz Jets for the AlphaT calcualtion
  std::vector<LorentzV> jets;
  CaloJetCollection::const_iterator ijet     = recocalojets->begin();
  CaloJetCollection::const_iterator ijetFast = recocalojetsFastJet->begin();
  CaloJetCollection::const_iterator jjet     = recocalojets->end(); 



  for( ; ijet != jjet; ijet++, ijetFast++ ) {
    if( flag == 1) break;
    // Do Some Jet selection!
    if( fabs(ijet->eta()) > etaJet_.at(0) ) continue;
    if( ijet->et() < minPtJet_.at(1) ) continue;

      if( fabs(ijetFast->eta()) < etaJet_.at(0) ){
      if( ijetFast->et() > minPtJet_.at(1) ) {
    // Add to HT
        htFast += ijetFast->et();
      }
     }      
    
    // Add to JetVector    
    LorentzV JetLVec(ijet->pt(),ijet->eta(),ijet->phi(),ijet->mass());
    jets.push_back( JetLVec );
    double aT = AlphaT()(jets);
    if(htFast > minHt_ && aT > minAlphaT_){
      // set flat to one so that we don't carry on looping though the jets
        flag = 1;
    }
  }



  if (flag==1) {
    for (reco::CaloJetCollection::const_iterator recocalojet = recocalojets->begin(); recocalojet!=jjet; recocalojet++) {
      if (recocalojet->et() > minPtJet_.at(0)) {
        ref = CaloJetRef(recocalojets,distance(recocalojets->begin(),recocalojet));
        // filterobject->addObject(TriggerJet,ref);
        n++;
      }
    }
  }
}// events with at least two jet

// filter decision
bool accept(n>0);

// put filter object into the Event
// iEvent.put(filterobject);

return accept;
}
