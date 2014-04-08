// -*- C++ -*-
//
// Package:    HLTCAWZTagFilter
// Class:      HLTCAWZTagFilter
// 
/**\class HLTCAWZTagFilter HLTCAWZTagFilter.cc UserCode/HLTCAWZTagFilter/plugins/HLTCAWZTagFilter.cc

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

#include "HLTrigger/JetMET/interface/HLTCAWZTagFilter.h"


using namespace std;
using namespace reco;
using namespace edm;

//
// constructors and destructor
//

HLTCAWZTagFilter::HLTCAWZTagFilter(const edm::ParameterSet& iConfig): HLTFilter(iConfig),
								      src_  (iConfig.getParameter<edm::InputTag>("src")),
								      pfsrc_ (iConfig.getParameter<edm::InputTag>("pfsrc")),
								   inputToken_ (consumes<reco::BasicJetCollection>(src_)),
								   inputPFToken_ (consumes<reco::PFJetCollection>(pfsrc_))
{
  minWMass_ = iConfig.getParameter<double>("minWMass");
  maxWMass_ = iConfig.getParameter<double>("maxWMass");
  massdropcut_ = iConfig.getParameter<double>("massdropcut");
  
}


HLTCAWZTagFilter::~HLTCAWZTagFilter()
{
}


void HLTCAWZTagFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions){
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<double>("maxWMass",130.);
  desc.add<double>("minWMass",60.);
  desc.add<double>("massdropcut",0.4);
  desc.add<edm::InputTag>("src",edm::InputTag("hltParticleFlow"));
  desc.add<edm::InputTag>("pfsrc",edm::InputTag("selectedPFJets"));
  desc.add<int>("triggerType",trigger::TriggerJet);
  descriptions.add("hltCA8WZTagFilter",desc);
}

// ------------ method called to for each event  ------------

bool HLTCAWZTagFilter::hltFilter( edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterobject) const
{

  // Get the input list of basic jets corresponding to the hard jets
  Handle<reco::BasicJetCollection> pBasicJets;
  iEvent.getByToken(inputToken_, pBasicJets);

 //get corresponding pf jets
  Handle<reco::PFJetCollection> pfJets;
  iEvent.getByToken(inputPFToken_, pfJets);


  //add filter object
  if(saveTags()){
    filterobject.addCollectionTag(pfsrc_);
  }

  //initialize the properties
  CAWZJetHelperUser helper( massdropcut_ );
  CATopJetProperties properties;

  // Now loop over the hard jets and do kinematic cuts
  reco::BasicJetCollection::const_iterator ihardJet = pBasicJets->begin(),
    ihardJetEnd = pBasicJets->end();
  reco::PFJetCollection::const_iterator ipfJet = pfJets->begin();
  bool pass = false;

  for ( ; ihardJet != ihardJetEnd; ++ihardJet, ++ipfJet ) {

    //if (ihardJet->pt() < 150) continue;

    // Get properties
    properties = helper( (reco::Jet&) *ihardJet );

    if (properties.nSubJets !=2 || properties.wMass < minWMass_ || properties.wMass > maxWMass_) continue;
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

 

//define this as a plug-in
DEFINE_FWK_MODULE(HLTCAWZTagFilter);
