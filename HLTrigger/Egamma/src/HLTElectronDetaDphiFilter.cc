/** \class HLTElectronDetaDphiFilter
 *
 * $Id: HLTElectronDetaDphiFilter.cc,v 1.7 2009/01/15 14:31:49 covarell Exp $ 
 *
 *  \author Alessio Ghezzi (Milano-Bicocca & CERN)
 *
 */

#include "HLTrigger/Egamma/interface/HLTElectronDetaDphiFilter.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/EgammaCandidates/interface/Electron.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/EgammaCandidates/interface/ElectronIsolationAssociation.h"

//
// constructors and destructor
//
HLTElectronDetaDphiFilter::HLTElectronDetaDphiFilter(const edm::ParameterSet& iConfig){
 
  candTag_ = iConfig.getParameter< edm::InputTag > ("candTag");
  DeltaEtaisoTag_ = iConfig.getParameter< edm::InputTag > ("isoTagDeltaEta");
  DeltaEtanonIsoTag_ = iConfig.getParameter< edm::InputTag > ("nonIsoTagDeltaEta");
  DeltaPhiisoTag_ = iConfig.getParameter< edm::InputTag > ("isoTagDeltaPhi");
  DeltaPhinonIsoTag_ = iConfig.getParameter< edm::InputTag > ("nonIsoTagDeltaPhi");

  DeltaEtacut_ = iConfig.getParameter<double> ("DeltaEtaCut");
  DeltaPhicut_ = iConfig.getParameter<double> ("DeltaPhiCut");
  ncandcut_  = iConfig.getParameter<int> ("ncandcut");
  doIsolated_ = iConfig.getParameter<bool> ("doIsolated");
   
  store_ = iConfig.getUntrackedParameter<bool> ("SaveTag",false) ;
  relaxed_ = iConfig.getUntrackedParameter<bool> ("relaxed",true) ;
  L1IsoCollTag_= iConfig.getParameter< edm::InputTag > ("L1IsoCand"); 
  L1NonIsoCollTag_= iConfig.getParameter< edm::InputTag > ("L1NonIsoCand"); 

  //register your products
  produces<trigger::TriggerFilterObjectWithRefs>();
}

HLTElectronDetaDphiFilter::~HLTElectronDetaDphiFilter(){}


// ------------ method called to produce the data  ------------
bool
HLTElectronDetaDphiFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  
 // The filter object
  using namespace trigger;
  std::auto_ptr<trigger::TriggerFilterObjectWithRefs> filterproduct (new trigger::TriggerFilterObjectWithRefs(path(),module()));
  if( store_ ){filterproduct->addCollectionTag(L1IsoCollTag_);}
  if( store_ && relaxed_){filterproduct->addCollectionTag(L1NonIsoCollTag_);}
  // Ref to Candidate object to be recorded in filter object
  edm::Ref<reco::ElectronCollection> ref;

  edm::Handle<trigger::TriggerFilterObjectWithRefs> PrevFilterOutput;

  iEvent.getByLabel (candTag_,PrevFilterOutput);

  std::vector<edm::Ref<reco::ElectronCollection> > elecands;
  PrevFilterOutput->getObjects(TriggerElectron, elecands);

  // retrieve Deta-Dphi association maps
  edm::Handle<reco::ElectronIsolationMap> depMapEta;
  iEvent.getByLabel (DeltaEtaisoTag_,depMapEta);
  edm::Handle<reco::ElectronIsolationMap> depNonIsoMapEta;
  if (!doIsolated_) iEvent.getByLabel (DeltaEtanonIsoTag_,depNonIsoMapEta);

  edm::Handle<reco::ElectronIsolationMap> depMapPhi;
  iEvent.getByLabel (DeltaPhiisoTag_,depMapPhi);
  edm::Handle<reco::ElectronIsolationMap> depNonIsoMapPhi;
  if (!doIsolated_) iEvent.getByLabel (DeltaPhinonIsoTag_,depNonIsoMapPhi);

  int n = 0;
  
  for (unsigned int i=0; i<elecands.size(); i++) {

    reco::ElectronRef eleref = elecands[i];
    
    reco::ElectronIsolationMap::const_iterator mapieta = (*depMapEta).find( eleref );
    if( mapieta==(*depMapEta).end() && !doIsolated_) mapieta = (*depNonIsoMapEta).find( eleref );

    reco::ElectronIsolationMap::const_iterator mapiphi = (*depMapPhi).find( eleref );
    if( mapiphi==(*depMapPhi).end() && !doIsolated_) mapiphi = (*depNonIsoMapPhi).find( eleref ); 
    
    float deltaeta = mapieta->val;
    float deltaphi = mapiphi->val;

    if( deltaeta < DeltaEtacut_  &&  deltaphi < DeltaPhicut_ ){
      n++;
      filterproduct->addObject(TriggerElectron, eleref);
    }

  }
  
  // filter decision
  bool accept(n>=ncandcut_);
  
  // put filter object into the Event
  iEvent.put(filterproduct);

   return accept;
}

