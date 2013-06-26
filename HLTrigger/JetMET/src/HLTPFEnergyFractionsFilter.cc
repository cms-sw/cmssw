/** \class HLTPFEnergyFractionsFilter
*
*
*  \author Srimanobhas Phat
*
*  Please see the description of this class in
*  "HLTrigger/JetMET/interface/HLTPFEnergyFractionsFilter.h"
*
*/

#include "HLTrigger/JetMET/interface/HLTPFEnergyFractionsFilter.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"

//
// constructors and destructor
//
HLTPFEnergyFractionsFilter::HLTPFEnergyFractionsFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig) 
{
  inputPFJetTag_         = iConfig.getParameter< edm::InputTag > ("inputPFJetTag");
  nJet_                  = iConfig.getParameter<unsigned int> ("nJet");
  min_CEEF_              = iConfig.getParameter<double> ("min_CEEF");
  max_CEEF_              = iConfig.getParameter<double> ("max_CEEF");
  min_NEEF_              = iConfig.getParameter<double> ("min_NEEF");
  max_NEEF_              = iConfig.getParameter<double> ("max_NEEF");
  min_CHEF_              = iConfig.getParameter<double> ("min_CHEF");
  max_CHEF_              = iConfig.getParameter<double> ("max_CHEF");
  min_NHEF_              = iConfig.getParameter<double> ("min_NHEF");
  max_NHEF_              = iConfig.getParameter<double> ("max_NHEF");
  triggerType_           = iConfig.getParameter<int> ("triggerType");
}

HLTPFEnergyFractionsFilter::~HLTPFEnergyFractionsFilter(){}

void 
HLTPFEnergyFractionsFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("inputPFJetTag",edm::InputTag("hltAntiKT5PFJets"));
  desc.add<unsigned int>("nJet",1);
  desc.add<double>("min_CEEF",-99.);
  desc.add<double>("max_CEEF",99.); 
  desc.add<double>("min_NEEF",-99.);
  desc.add<double>("max_NEEF",99.);
  desc.add<double>("min_CHEF",-99.);
  desc.add<double>("max_CHEF",99.);
  desc.add<double>("min_NHEF",-99.);
  desc.add<double>("max_NHEF",99.);
  desc.add<int>("triggerType",trigger::TriggerJet);
  descriptions.add("hltPFEnergyFractionsFilter",desc);
}

// ------------ method called to produce the data  ------------
bool
HLTPFEnergyFractionsFilter::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct)
{
  using namespace std;
  using namespace edm;
  using namespace reco;
  using namespace trigger;
  
  // The filter object
  if (saveTags()) filterproduct.addCollectionTag(inputPFJetTag_);
  
  // PFJets
  edm::Handle<PFJetCollection> recopfjets;
  iEvent.getByLabel(inputPFJetTag_,recopfjets);

  //Checking
  bool accept(false); 
  
  if(recopfjets->size() >= nJet_){
    accept = true;
    unsigned int countJet(0);
    //PF information
    PFJetCollection::const_iterator i (recopfjets->begin());
    for(; i != recopfjets->end(); ++i ){
      if(countJet>=nJet_) break;
      //
      if(i->chargedEmEnergyFraction()<min_CEEF_) accept = false;
      if(i->chargedEmEnergyFraction()>max_CEEF_) accept = false;
      //
      if(i->neutralEmEnergyFraction()<min_NEEF_) accept = false;
      if(i->neutralEmEnergyFraction()>max_NEEF_) accept = false;
      //
      if(i->chargedHadronEnergyFraction()<min_CHEF_) accept = false;
      if(i->chargedHadronEnergyFraction()>max_CHEF_) accept = false;
      //
      if(i->neutralHadronEnergyFraction()<min_NHEF_) accept = false;
      if(i->neutralHadronEnergyFraction()>max_NHEF_) accept = false;
      //
      if(accept==false) break;
      countJet++;
    }
    
    //Store NJet_ jets
    if(accept==true){ 
      countJet = 0; 
      PFJetCollection::const_iterator i (recopfjets->begin());
      for(; i != recopfjets->end(); ++i ){
	if(countJet>=nJet_) break;
	filterproduct.addObject(triggerType_,PFJetRef(recopfjets,distance(recopfjets->begin(),i)));
	countJet++;
      }
    }
  }// End of (recopfjets->size() >= nJet_)
  
  return accept;
}
