// -*- C++ -*-
//
// Package:    HLTCATopTagFilter
// Class:      HLTCATopTagFilter
// 
/**\class HLTCATopTagFilter HLTCATopTagFilter.cc UserCode/HLTCATopTagFilter/plugins/HLTCATopTagFilter.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  clint richardson
//         Created:  Sat, 5 Jul 2014 20:23:57 GMT
// $Id$
//
//

#include "HLTrigger/JetMET/interface/HLTJetMassFilter.h"
#include <typeinfo>

using namespace std;
using namespace reco;
using namespace edm;

//
// constructors and destructor
//

HLTJetMassFilter::HLTJetMassFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig),
								       src_ (iConfig.getParameter<edm::InputTag>("src")),
								       inputPFToken_ (consumes<reco::PFJetCollection>(src_))
{
  minJetMass_ = iConfig.getParameter<double>("minJetMass");

}

HLTJetMassFilter::~HLTJetMassFilter(){}

void HLTJetMassFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions){
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("src",edm::InputTag("selectedPFJets"));
  desc.add<double>("minJetMass",30.);
  desc.add<int>("triggerType",trigger::TriggerJet);
  descriptions.add("hltJetMassFilter",desc);
}


bool HLTJetMassFilter::hltFilter( edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs& filterobject) const
{

  //read in jet collection
  Handle<reco::PFJetCollection> Jets;
  iEvent.getByToken(inputPFToken_, Jets);

  //add filter object
  if(saveTags()){
    filterobject.addCollectionTag(src_);
  }

  //iterate over jets in event and return pass if at least one passes jet mass cut
  reco::PFJetCollection::const_iterator iJet = Jets->begin(), iEndJet = Jets->end();
  bool pass = false;

  for( ; iJet != iEndJet; ++iJet){
    
    float mass = iJet->mass();

    if(mass>minJetMass_){
      //update pass info
      pass=true;
      //get ref to jet and add
      reco::PFJetRef ref = reco::PFJetRef(Jets,distance(Jets->begin(),iJet));
      filterobject.addObject(trigger::TriggerJet,ref);
    }

  }

  return pass;

}



//define as plugin
DEFINE_FWK_MODULE(HLTJetMassFilter);
