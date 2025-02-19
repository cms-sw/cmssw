/** \class HLTEgammaAllCombMassFilter
 *
 * $Id: HLTEgammaAllCombMassFilter.cc,
 *
 *  \author Chris Tully (Princeton)
 *
 */

#include "HLTrigger/Egamma/interface/HLTEgammaAllCombMassFilter.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"

//
// constructors and destructor
//
HLTEgammaAllCombMassFilter::HLTEgammaAllCombMassFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig) 
{
  firstLegLastFilterTag_ = iConfig.getParameter<edm::InputTag>("firstLegLastFilter");
  secondLegLastFilterTag_= iConfig.getParameter<edm::InputTag>("secondLegLastFilter");
  minMass_ = iConfig.getParameter<double> ("minMass");
}

HLTEgammaAllCombMassFilter::~HLTEgammaAllCombMassFilter(){}


// ------------ method called to produce the data  ------------
bool HLTEgammaAllCombMassFilter::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct)
{
  //right, issue 1, we dont know if this is a TriggerElectron, TriggerPhoton, TriggerCluster (should never be a TriggerCluster btw as that implies the 4-vectors are not stored in AOD)

  //trigger::TriggerObjectType firstLegTrigType;
  std::vector<math::XYZTLorentzVector> firstLegP4s;
  
  //trigger::TriggerObjectType secondLegTrigType;
  std::vector<math::XYZTLorentzVector> secondLegP4s;

  math::XYZTLorentzVector pairP4;

  getP4OfLegCands(iEvent,firstLegLastFilterTag_,firstLegP4s);
  getP4OfLegCands(iEvent,secondLegLastFilterTag_,secondLegP4s);

  bool accept=false;
  for(size_t firstLegNr=0;firstLegNr<firstLegP4s.size();firstLegNr++){ 
    for(size_t secondLegNr=0;secondLegNr<secondLegP4s.size();secondLegNr++){
      math::XYZTLorentzVector pairP4 = firstLegP4s[firstLegNr] + secondLegP4s[secondLegNr];
      double mass = pairP4.M();
      if(mass>=minMass_) accept=true;
    }
  }
  for(size_t firstLegNr=0;firstLegNr<firstLegP4s.size();firstLegNr++){ 
    for(size_t secondLegNr=0;secondLegNr<firstLegP4s.size();secondLegNr++){
      math::XYZTLorentzVector pairP4 = firstLegP4s[firstLegNr] + firstLegP4s[secondLegNr];
      double mass = pairP4.M();
      if(mass>=minMass_) accept=true;
    }
  }
  for(size_t firstLegNr=0;firstLegNr<secondLegP4s.size();firstLegNr++){ 
    for(size_t secondLegNr=0;secondLegNr<secondLegP4s.size();secondLegNr++){
      math::XYZTLorentzVector pairP4 = secondLegP4s[firstLegNr] + secondLegP4s[secondLegNr];
      double mass = pairP4.M();
      if(mass>=minMass_) accept=true;
    }
  }
  
  return accept;
}

void  HLTEgammaAllCombMassFilter::getP4OfLegCands(const edm::Event& iEvent,edm::InputTag filterTag,std::vector<math::XYZTLorentzVector>& p4s)
{ 
  edm::Handle<trigger::TriggerFilterObjectWithRefs> filterOutput;
  iEvent.getByLabel(filterTag,filterOutput);
  
  //its easier on the if statement flow if I try everything at once, shouldnt add to timing
  std::vector<edm::Ref<reco::RecoEcalCandidateCollection> > phoCands;
  filterOutput->getObjects(trigger::TriggerPhoton,phoCands);
  std::vector<edm::Ref<reco::RecoEcalCandidateCollection> > clusCands;
  filterOutput->getObjects(trigger::TriggerCluster,clusCands);
  std::vector<edm::Ref<reco::ElectronCollection> > eleCands;
  filterOutput->getObjects(trigger::TriggerElectron,eleCands);

  if(!phoCands.empty()){ //its photons
    for(size_t candNr=0;candNr<phoCands.size();candNr++){
      p4s.push_back(phoCands[candNr]->p4());
    }
  }else if(!clusCands.empty()){ //try trigger cluster (should never be this, at the time of writing (17/1/11) this would indicate an error)
    for(size_t candNr=0;candNr<clusCands.size();candNr++){
      p4s.push_back(clusCands[candNr]->p4());
    }
  }else if(!eleCands.empty()){
    for(size_t candNr=0;candNr<eleCands.size();candNr++){
      p4s.push_back(eleCands[candNr]->p4());
    }
  }
}
