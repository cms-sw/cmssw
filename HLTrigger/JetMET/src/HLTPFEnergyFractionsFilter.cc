/** \class HLTPFEnergyFractionsFilter
*
*
*  \author Srimanobhas N.
*
*/

#include "HLTrigger/JetMET/interface/HLTPFEnergyFractionsFilter.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/JetReco/interface/CaloJetCollection.h"
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
  inputPFJetTag_           = iConfig.getParameter< edm::InputTag > ("inputPFJetTag");
  inputCaloJetTag_         = iConfig.getParameter< edm::InputTag > ("inputCaloJetTag");
  nJet_                  = iConfig.getParameter<unsigned int> ("nJet");
  min_CEEF_              = iConfig.getParameter<double> ("min_CEEF");
  max_CEEF_              = iConfig.getParameter<double> ("max_CEEF");
  min_NEEF_              = iConfig.getParameter<double> ("min_NEEF");
  max_NEEF_              = iConfig.getParameter<double> ("max_NEEF");
  min_CHEF_              = iConfig.getParameter<double> ("min_CHEF");
  max_CHEF_              = iConfig.getParameter<double> ("max_CHEF");
  min_NHEF_              = iConfig.getParameter<double> ("min_NHEF");
  max_NHEF_              = iConfig.getParameter<double> ("max_NHEF");
}

HLTPFEnergyFractionsFilter::~HLTPFEnergyFractionsFilter(){}

void 
HLTPFEnergyFractionsFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("inputPFJetTag",edm::InputTag("hltAntiKT5PFJets"));
  desc.add<edm::InputTag>("inputCaloJetTag",edm::InputTag("hltAntiKT5ConvPFJets"));
  //
  desc.add<bool>("saveTags",false);
  //
  desc.add<unsigned int>("nJet",1);
  desc.add<double>("min_CEEF",-99.);
  desc.add<double>("max_CEEF",99.); 
  desc.add<double>("min_NEEF",-99.);
  desc.add<double>("max_NEEF",99.);
  desc.add<double>("min_CHEF",-99.);
  desc.add<double>("max_CHEF",99.);
  desc.add<double>("min_NHEF",-99.);
  desc.add<double>("max_NHEF",99.);
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
  if (saveTags()) filterproduct.addCollectionTag(inputCaloJetTag_);

  // CaloJets
  edm::Handle<CaloJetCollection> recocalojets;
  iEvent.getByLabel(inputCaloJetTag_,recocalojets);
  // PFJets
  edm::Handle<PFJetCollection> recopfjets;
  iEvent.getByLabel(inputPFJetTag_,recopfjets);

  //Checking
  int n(0); 
  CaloJetRef JetRef1;

  if(recopfjets->size() >= nJet_){
    unsigned int countJet(0); 
    double pf1Pt=0., pf1Eta=0., pf1Phi=0.;
    double calo1Pt=0., calo1Eta=0., calo1Phi=0.;
    
    //PF information
    for(PFJetCollection::const_iterator i = recopfjets->begin(); i != recopfjets->end(); ++i ){
      if(countJet<nJet_){
	if(i->chargedEmEnergyFraction()<min_CEEF_) n = -1;
	if(i->chargedEmEnergyFraction()>max_CEEF_) n = -1;
	//
	if(i->neutralEmEnergyFraction()<min_NEEF_) n = -1;
	if(i->neutralEmEnergyFraction()>max_NEEF_) n = -1;
	//
	if(i->chargedHadronEnergyFraction()<min_CHEF_) n = -1;
	if(i->chargedHadronEnergyFraction()>max_CHEF_) n = -1;
	//
	if(i->neutralHadronEnergyFraction()<min_NHEF_) n = -1;
	if(i->neutralHadronEnergyFraction()>max_NHEF_) n = -1;
	//
	if(countJet==0){
	  pf1Pt   = i->pt();
	  pf1Eta  = i->eta();
	  pf1Phi  = i->phi();
	}
      }
      countJet++;
      if(countJet>=nJet_) break;
    }
    //Calo information
    countJet=0;
    for (CaloJetCollection::const_iterator j = recocalojets->begin();
	 j != recocalojets->end(); j++) {
      if(countJet==0){
	JetRef1 = CaloJetRef(recocalojets,distance(recocalojets->begin(),j)); 
	calo1Pt   = j->pt();
	calo1Eta  = j->eta();
	calo1Phi  = j->phi();
	break;
      }
    }
    
    //x-check pfjet<->conv calojet
    if(fabs(pf1Pt-calo1Pt)>0.001)   n = -1;
    if(fabs(pf1Eta-calo1Eta)>0.001) n = -1;
    if(fabs(pf1Phi-calo1Phi)>0.001) n = -1;
    if(n==0) n++;
  }
  
  //Store only 1st pt jet which pass conditions
  if(n>0){
    filterproduct.addObject(TriggerJet,JetRef1);
  }
  
  // filter decision
  bool accept(n>0); 
  
  return accept;
}
