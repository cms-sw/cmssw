// -*- C++ -*-
//
// Package:    BasicToPFJet
// Class:      BasicToPFJet
// 
/**\class BasicToPFJet BasicToPFJet.cc UserCode/BasicToPFJet/plugins/BasicToPFJet.cc

 Description: converts reco::BasicJets to reco::PFJets and adds the new PFJetCollection to the event. Originally designed
              to be a work around for a way to store reco::BasicJets at HLT level

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  clint richardson
//         Created:  Thu, 6 Mar 2014 12:00:00 GMT
//
//
// system include files
#include <memory>
#include <vector>
#include <sstream>

// user include files
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/JetReco/interface/CATopJetTagInfo.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"

//include header file
#include "RecoJets/JetProducers/plugins/BasicToPFJet.h"

BasicToPFJet::BasicToPFJet(const edm::ParameterSet& PSet) :
  src_ (PSet.getParameter<edm::InputTag>("src")),
  inputToken_ (consumes<reco::BasicJetCollection>(src_))
{
  produces<reco::PFJetCollection>();
}

BasicToPFJet::~BasicToPFJet(){}


void BasicToPFJet::fillDescriptions(edm::ConfigurationDescriptions& descriptions){
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src",edm::InputTag(""));
  descriptions.add("BasicToPFJet",desc);
}

void BasicToPFJet::produce( edm::Event& Event, const edm::EventSetup& EventSetup){

  //first get the basic jet collection
  edm::Handle<reco::BasicJetCollection> BasicJetColl;
  Event.getByToken(inputToken_, BasicJetColl);

  //now make the new pf jet collection
  std::auto_ptr<reco::PFJetCollection> PFJetColl(new reco::PFJetCollection);
  //reco::PFJetCollection* PFJetColl = new reco::PFJetCollection;
  //make the 'specific'
  reco::PFJet::Specific specific;

  //now get iterator
  reco::BasicJetCollection::const_iterator i = BasicJetColl->begin();

  //loop over basic jets and convert them to pfjets
  for(; i!=BasicJetColl->end(); i++){
    reco::PFJet pfjet(i->p4(),i->vertex(),specific);
    PFJetColl->push_back(pfjet);
  }

  //std::auto_ptr<reco::PFJetCollection> selectedPFJets(PFJetColl);
  Event.put(PFJetColl);
}

 
//define as plug-in for the framework
DEFINE_FWK_MODULE(BasicToPFJet);
