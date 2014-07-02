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
// Original Author:  dylan rankin
//         Created:  Wed, 17 Jul 2013 22:11:30 GMT
// $Id$
//
//

#include "HLTrigger/JetMET/interface/HLTCATopTagFilter.h"
#include <typeinfo>

using namespace std;
using namespace reco;
using namespace edm;

//
// constructors and destructor
//

HLTCATopTagFilter::HLTCATopTagFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig),  
								      src_  (iConfig.getParameter<edm::InputTag>("src")),
								      pfsrc_ (iConfig.getParameter<edm::InputTag>("pfsrc")),
								      inputToken_ (consumes<reco::BasicJetCollection>(src_)),
								      inputPFToken_ (consumes<reco::PFJetCollection>(pfsrc_))
{
  TopMass_ = iConfig.getParameter<double>("TopMass");
  minTopMass_ = iConfig.getParameter<double>("minTopMass");
  maxTopMass_ = iConfig.getParameter<double>("maxTopMass");
  minMinMass_ = iConfig.getParameter<double>("minMinMass");
}


HLTCATopTagFilter::~HLTCATopTagFilter(){}


void HLTCATopTagFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions){
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<double>("TopMass",171.);
  desc.add<double>("maxTopMass",230.);
  desc.add<double>("minTopMass",140.);
  desc.add<double>("minMinMass",50.);
  desc.add<edm::InputTag>("src",edm::InputTag("hltParticleFlow"));
  desc.add<edm::InputTag>("pfsrc",edm::InputTag("selectedPFJets"));
  desc.add<int>("triggerType",trigger::TriggerJet);
  descriptions.add("hltCA8TopTagFilter",desc);
}
// ------------ method called to for each event  ------------

bool HLTCATopTagFilter::hltFilter( edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterobject) const
{


  //get basic jets
  Handle<reco::BasicJetCollection > pBasicJets;
  iEvent.getByToken(inputToken_, pBasicJets);

  //get corresponding pf jets
  Handle<reco::PFJetCollection> pfJets;
  iEvent.getByToken(inputPFToken_, pfJets);

  //add filter object
  if(saveTags()){
    filterobject.addCollectionTag(pfsrc_);
  }

  //initialize the properties
  CATopJetHelperUser helper( TopMass_);
  CATopJetProperties properties;

  // Now loop over the hard jets and do kinematic cuts
  reco::BasicJetCollection::const_iterator ihardJet = pBasicJets->begin(),
    ihardJetEnd = pBasicJets->end();
  reco::PFJetCollection::const_iterator ipfJet = pfJets->begin();
  bool pass = false;
  
  for ( ; ihardJet != ihardJetEnd; ++ihardJet, ++ipfJet ) {

    //if (ihardJet->pt() < 350) continue;

    // Get properties
    properties = helper( (reco::Jet&) *ihardJet );

    if (properties.nSubJets < 3 ||properties.minMass < minMinMass_ || properties.topMass < minTopMass_ || properties.topMass > maxTopMass_) continue;
    else {
      // Get a ref to the hard jet
      reco::PFJetRef ref = reco::PFJetRef(pfJets,distance(pfJets->begin(),ipfJet));
      //add ref to event
      filterobject.addObject(trigger::TriggerJet,ref);
      pass = true;
    }

  }// end loop over hard jets




  return pass;
}
// ------------ method called once each job just before starting event loop  ------------

 

//define this as a plug-in
DEFINE_FWK_MODULE(HLTCATopTagFilter);
