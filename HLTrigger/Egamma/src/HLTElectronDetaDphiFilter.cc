/** \class HLTElectronDetaDphiFilter
 *
 * $Id: HLTElectronDetaDphiFilter.cc,v 1.1 2008/03/03 12:51:39 ghezzi Exp $ 
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


//
// constructors and destructor
//
HLTElectronDetaDphiFilter::HLTElectronDetaDphiFilter(const edm::ParameterSet& iConfig){
  candTag_ = iConfig.getParameter< edm::InputTag > ("candTag");
  DeltaEtacut_ =  iConfig.getParameter<double> ("DeltaEtaCut");
  DeltaPhicut_ =  iConfig.getParameter<double> ("DeltaPhiCut");
  ncandcut_  = iConfig.getParameter<int> ("ncandcut");
  //doIsolated_ = iConfig.getParameter<bool> ("doIsolated");

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
  // Ref to Candidate object to be recorded in filter object
  edm::Ref<reco::ElectronCollection> ref;


  edm::Handle<trigger::TriggerFilterObjectWithRefs> PrevFilterOutput;

  iEvent.getByLabel (candTag_,PrevFilterOutput);

  std::vector<edm::Ref<reco::ElectronCollection> > elecands;
  PrevFilterOutput->getObjects(TriggerElectron, elecands);

  // look at all electrons,  check cuts and add to filter object
  int n = 0;
  
  for (unsigned int i=0; i<elecands.size(); i++) {

    reco::ElectronRef eleref = elecands[i];
    const reco::SuperClusterRef theClus = eleref->superCluster();
    const math::XYZVector trackMom =  eleref->track()->momentum();
    float deltaphi=fabs(trackMom.phi()-theClus->phi());
    if(deltaphi>6.283185308) deltaphi-=6.283185308;
    if(deltaphi>3.141592654) deltaphi=6.283185308-deltaphi;

    if( fabs(trackMom.eta()-theClus->eta()) < DeltaEtacut_  &&   deltaphi < DeltaPhicut_ ){
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

