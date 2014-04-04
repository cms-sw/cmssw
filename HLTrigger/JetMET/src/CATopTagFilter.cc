// -*- C++ -*-
//
// Package:    CATopTagFilter
// Class:      CATopTagFilter
// 
/**\class CATopTagFilter CATopTagFilter.cc UserCode/CATopTagFilter/plugins/CATopTagFilter.cc

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

#include "../interface/CATopTagFilter.h"
#include <typeinfo>

using namespace std;
using namespace reco;
using namespace edm;

//
// constructors and destructor
//

CATopTagFilter::CATopTagFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig),  
								      src_  (iConfig.getParameter<edm::InputTag>("src")),
								      pfsrc_ (iConfig.getParameter<edm::InputTag>("pfsrc")),
								      inputToken_ (consumes<reco::BasicJetCollection>(src_)),
								      inputPFToken_ (consumes<reco::PFJetCollection>(pfsrc_))
{
  if ( iConfig.exists("TopMass") ) TopMass_ = iConfig.getParameter<double>("TopMass");
  else TopMass_ = 171.;
  if ( iConfig.exists("minTopMass") ) minTopMass_ = iConfig.getParameter<double>("minTopMass");
  else minTopMass_ = -1;
  if ( iConfig.exists("maxTopMass") ) maxTopMass_ = iConfig.getParameter<double>("maxTopMass");
  else maxTopMass_ = 999999;
  if ( iConfig.exists("WMass") ) WMass_ = iConfig.getParameter<double>("WMass");
  else WMass_ = 80.4;
  if ( iConfig.exists("minWMass") ) minWMass_ = iConfig.getParameter<double>("minWMass");
  else minWMass_ = -1;
  if ( iConfig.exists("maxWMass") ) maxWMass_ = iConfig.getParameter<double>("maxWMass");
  else maxWMass_ = 999999;
  if ( iConfig.exists("minMinMass") ) minMinMass_ = iConfig.getParameter<double>("minMinMass");
  else minMinMass_ = -1;
  if ( iConfig.exists("maxMinMass") ) maxMinMass_ = iConfig.getParameter<double>("maxMinMass");
  else maxMinMass_ = 999999;
}


CATopTagFilter::~CATopTagFilter(){}


void CATopTagFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions){
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<double>("maxTopMass",230.);
  desc.add<double>("minMinMass",50.);
  desc.add<double>("minTopMass",140.);
  desc.add<edm::InputTag>("src",edm::InputTag("hltParticleFlow"));
  desc.add<edm::InputTag>("pfsrc",edm::InputTag("selectedPFJets"));
  desc.add<int>("triggerType",trigger::TriggerJet);
  descriptions.add("hltCA8TopTagFilter",desc);
}
// ------------ method called to for each event  ------------

bool CATopTagFilter::hltFilter( edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterobject) const
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
  CATopJetHelperUser helper( TopMass_, WMass_ );
  CATopJetProperties properties;

  // Now loop over the hard jets and do kinematic cuts
  reco::BasicJetCollection::const_iterator ihardJet = pBasicJets->begin(),
    ihardJetEnd = pBasicJets->end();
  reco::PFJetCollection::const_iterator ipfJet = pfJets->begin();
  bool pass = false;
  
  for ( ; ihardJet != ihardJetEnd; ++ihardJet, ++ipfJet ) {

    if (ihardJet->pt() < 350) continue;

    // Get properties
    properties = helper( (reco::Jet&) *ihardJet );

    if (properties.minMass < minMinMass_ || properties.minMass > maxMinMass_ || properties.wMass < minWMass_ || properties.wMass > maxWMass_ || properties.topMass < minTopMass_ || properties.topMass > maxTopMass_) continue;
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
DEFINE_FWK_MODULE(CATopTagFilter);
