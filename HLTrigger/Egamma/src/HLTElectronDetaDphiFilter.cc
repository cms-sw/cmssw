/** \class HLTElectronDetaDphiFilter
 *
 * $Id: HLTElectronDetaDphiFilter.cc,v 1.2 2008/03/03 16:34:30 ghezzi Exp $ 
 *
 *  \author Alessio Ghezzi (Milano-Bicocca & CERN)
 *
 */

#include "HLTrigger/Egamma/interface/HLTElectronDetaDphiFilter.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/EgammaCandidates/interface/Electron.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"

#include "CondFormats/DataRecord/interface/BeamSpotObjectsRcd.h"//needed?
#include "CondFormats/BeamSpotObjects/interface/BeamSpotObjects.h"//needed?

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Math/interface/Point3D.h"
//
// constructors and destructor
//
HLTElectronDetaDphiFilter::HLTElectronDetaDphiFilter(const edm::ParameterSet& iConfig){
  candTag_ = iConfig.getParameter< edm::InputTag > ("candTag");
  DeltaEtacut_ =  iConfig.getParameter<double> ("DeltaEtaCut");
  DeltaPhicut_ =  iConfig.getParameter<double> ("DeltaPhiCut");
  ncandcut_  = iConfig.getParameter<int> ("ncandcut");
  //doIsolated_ = iConfig.getParameter<bool> ("doIsolated");
  BSProducer_ = iConfig.getParameter<edm::InputTag>("BSProducer");
  
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

  edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
  // iEvent.getByType(recoBeamSpotHandle);
  iEvent.getByLabel(BSProducer_,recoBeamSpotHandle);
  // gets its position
  const reco::BeamSpot::Point& BSPosition = recoBeamSpotHandle->position(); 
  // look at all electrons,  check cuts and add to filter object
  int n = 0;
  
  for (unsigned int i=0; i<elecands.size(); i++) {

    reco::ElectronRef eleref = elecands[i];
    const reco::SuperClusterRef theClus = eleref->superCluster();
    const math::XYZVector trackMom =  eleref->track()->momentum();
    
    math::XYZPoint SCcorrPosition(theClus->x()-BSPosition.x(), theClus->y()-BSPosition.y() , theClus->z()-eleref->track()->vz() );
    float deltaeta = SCcorrPosition.eta()-eleref->track()->eta();
    
    float deltaphi=fabs(trackMom.phi()-SCcorrPosition.phi());
    //FIX ME needs to be corrected for the curvature
    if(deltaphi>6.283185308) deltaphi-=6.283185308;
    if(deltaphi>3.141592654) deltaphi=6.283185308-deltaphi;

    if( fabs(deltaeta) < DeltaEtacut_  &&   deltaphi < DeltaPhicut_ ){
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

