/** \class HLTEgammaDoubleLegCombFilter
 *
 * $Id: HLTEgammaDoubleLegCombFilter.cc,
 *
 *  \author Sam Harper (RAL)
 *
 */

#include "HLTrigger/Egamma/interface/HLTEgammaDoubleLegCombFilter.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "DataFormats/EgammaReco/interface/SuperCluster.h"

//
// constructors and destructor
//
HLTEgammaDoubleLegCombFilter::HLTEgammaDoubleLegCombFilter(const edm::ParameterSet& iConfig)
{
  firstLegLastFilterTag_ = iConfig.getParameter<edm::InputTag>("firstLegLastFilter");
  secondLegLastFilterTag_= iConfig.getParameter<edm::InputTag>("secondLegLastFilter");
  nrRequiredFirstLeg_ = iConfig.getParameter<int> ("nrRequiredFirstLeg");
  nrRequiredSecondLeg_ = iConfig.getParameter<int> ("nrRequiredSecondLeg");
  maxMatchDR_ = iConfig.getParameter<double> ("maxMatchDR");
  
  //register your products
  produces<trigger::TriggerFilterObjectWithRefs>();
}

HLTEgammaDoubleLegCombFilter::~HLTEgammaDoubleLegCombFilter(){}


// ------------ method called to produce the data  ------------
bool HLTEgammaDoubleLegCombFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  std::auto_ptr<trigger::TriggerFilterObjectWithRefs> filterproduct (new trigger::TriggerFilterObjectWithRefs(path(),module())); //empty filter product

  //right, issue 1, we dont know if this is a TriggerElectron, TriggerPhoton, TriggerCluster (should never be a TriggerCluster btw as that implies the 4-vectors are not stored in AOD)

  

  //trigger::TriggerObjectType firstLegTrigType;
  std::vector<math::XYZPoint> firstLegP3s;
  
  //trigger::TriggerObjectType secondLegTrigType;
  std::vector<math::XYZPoint> secondLegP3s;

  getP3OfLegCands(iEvent,firstLegLastFilterTag_,firstLegP3s);
  getP3OfLegCands(iEvent,secondLegLastFilterTag_,secondLegP3s);

  std::vector<std::pair<int,int> > matchedCands; 
  matchCands(firstLegP3s,secondLegP3s,matchedCands);


  int nr1stLegOnly=0;
  int nr2ndLegOnly=0;
  int nrBoth=0;;
  
  for(size_t candNr=0;candNr<matchedCands.size();candNr++){
    if(matchedCands[candNr].first>=0){ //we found a first leg cand
      if(matchedCands[candNr].second>=0) nrBoth++;//we also found a second leg cand
      else nr1stLegOnly++; //we didnt find a second leg cand
    }else if(matchedCands[candNr].second>=0) nr2ndLegOnly++; //we found a second leg cand but we didnt find a first leg
    
  }
  
  bool accept=false;
  if(nr1stLegOnly>=nrRequiredFirstLeg_ && nr2ndLegOnly>=nrRequiredSecondLeg_) accept=true;
  else{ 
    int nrNeededFirstLeg = std::max(0,nrRequiredFirstLeg_ - nr1stLegOnly);
    int nrNeededSecondLeg = std::max(0,nrRequiredSecondLeg_ - nr2ndLegOnly);
    
    if(nrBoth >= nrNeededFirstLeg + nrNeededSecondLeg) accept = true;
  }
  
  //put filter object into the Event
  iEvent.put(filterproduct);

  return accept;
}

//-1 if no match is found
void  HLTEgammaDoubleLegCombFilter::matchCands(const std::vector<math::XYZPoint>& firstLegP3s,const std::vector<math::XYZPoint>& secondLegP3s,std::vector<std::pair<int,int> >&matchedCands)
{
  std::vector<size_t> matched2ndLegs;
  for(size_t firstLegNr=0;firstLegNr<firstLegP3s.size();firstLegNr++){ 
    int matchedNr = -1;
    for(size_t secondLegNr=0;secondLegNr<secondLegP3s.size() && matchedNr==-1;secondLegNr++){
      if(reco::deltaR2(firstLegP3s[firstLegNr],secondLegP3s[secondLegNr])<maxMatchDR_*maxMatchDR_) matchedNr=secondLegNr;
    }
    matchedCands.push_back(std::make_pair(firstLegNr,matchedNr));
    if(matchedNr>=0) matched2ndLegs.push_back(static_cast<size_t>(matchedNr));
  }
  std::sort(matched2ndLegs.begin(),matched2ndLegs.end());
  
  for(size_t secondLegNr=0;secondLegNr<secondLegP3s.size();secondLegNr++){
    if(!std::binary_search(matched2ndLegs.begin(),matched2ndLegs.end(),secondLegNr)){ //wasnt matched already
      matchedCands.push_back(std::make_pair<int,int>(-1,secondLegNr));
    }
  }
}

//we use position and p3 interchangably here, we only use eta/phi so its alright
void  HLTEgammaDoubleLegCombFilter::getP3OfLegCands(const edm::Event& iEvent,edm::InputTag filterTag,std::vector<math::XYZPoint>& p3s)
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
  std::vector<edm::Ref<reco::CaloJetCollection> > jetCands;
  filterOutput->getObjects(trigger::TriggerJet,jetCands);
 
  if(!phoCands.empty()){ //its photons
    for(size_t candNr=0;candNr<phoCands.size();candNr++){
      p3s.push_back(phoCands[candNr]->superCluster()->position());
    }
  }else if(!clusCands.empty()){ //try trigger cluster (should never be this, at the time of writing (17/1/11) this would indicate an error)
    for(size_t candNr=0;candNr<clusCands.size();candNr++){
      p3s.push_back(clusCands[candNr]->superCluster()->position());
    }
  }else if(!eleCands.empty()){
    for(size_t candNr=0;candNr<eleCands.size();candNr++){
      p3s.push_back(eleCands[candNr]->superCluster()->position());
    }
  }else if(!jetCands.empty()){
    for(size_t candNr=0;candNr<jetCands.size();candNr++){
      math::XYZPoint p3;
      p3.SetX(jetCands[candNr]->p4().x());
      p3.SetY(jetCands[candNr]->p4().y());
      p3.SetZ(jetCands[candNr]->p4().z());
      p3s.push_back(p3);
    }
  }
}
