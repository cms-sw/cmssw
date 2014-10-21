/** \class HLTElectronEoverpFilterRegional
 *
 *  \author Monica Vazquez Acosta (CERN)
 *
 *
 */

#include "HLTrigger/Egamma/interface/HLTElectronEoverpFilterRegional.h"

#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"

//
// constructors and destructor
//
HLTElectronEoverpFilterRegional::HLTElectronEoverpFilterRegional(const edm::ParameterSet& iConfig) : HLTFilter(iConfig)
{
   candTag_ = iConfig.getParameter< edm::InputTag > ("candTag");
   electronIsolatedProducer_ = iConfig.getParameter< edm::InputTag > ("electronIsolatedProducer");
   electronNonIsolatedProducer_ = iConfig.getParameter< edm::InputTag > ("electronNonIsolatedProducer");
   eoverpbarrelcut_  = iConfig.getParameter<double> ("eoverpbarrelcut");
   eoverpendcapcut_  = iConfig.getParameter<double> ("eoverpendcapcut");
   ncandcut_  = iConfig.getParameter<int> ("ncandcut");
   doIsolated_  = iConfig.getParameter<bool> ("doIsolated");
   candToken_ =  consumes<trigger::TriggerFilterObjectWithRefs>(candTag_);
   electronIsolatedToken_ = consumes<reco::ElectronCollection>(electronIsolatedProducer_);
   if(!doIsolated_) electronNonIsolatedToken_ = consumes<reco::ElectronCollection>(electronNonIsolatedProducer_);
}

HLTElectronEoverpFilterRegional::~HLTElectronEoverpFilterRegional(){}

void
HLTElectronEoverpFilterRegional::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("candTag",edm::InputTag("hltElectronPixelMatchFilter"));
  desc.add<edm::InputTag>("electronIsolatedProducer",edm::InputTag("pixelMatchElectronsForHLT"));
  desc.add<edm::InputTag>("electronNonIsolatedProducer",edm::InputTag("pixelMatchElectronsForHLT"));
  desc.add<double>("eoverpbarrelcut",1.5);
  desc.add<double>("eoverpendcapcut",2.45);
  desc.add<int>("ncandcut",1);
  desc.add<bool>("doIsolated",true);
  descriptions.add("hltElectronEoverpFilter",desc);
}

// ------------ method called to produce the data  ------------
bool
HLTElectronEoverpFilterRegional::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct) const
{

  // The filter object
  using namespace trigger;
  if (saveTags()) {
    filterproduct.addCollectionTag(electronIsolatedProducer_);
    if (not doIsolated_) filterproduct.addCollectionTag(electronNonIsolatedProducer_);
  }

  edm::Handle<trigger::TriggerFilterObjectWithRefs> PrevFilterOutput;
  iEvent.getByToken (candToken_,PrevFilterOutput);

  std::vector<edm::Ref<reco::RecoEcalCandidateCollection> > recoecalcands;
  PrevFilterOutput->getObjects(TriggerCluster, recoecalcands);
  if(recoecalcands.empty()) PrevFilterOutput->getObjects(TriggerPhoton, recoecalcands);
   // Get the HLT electrons from EgammaHLTPixelMatchElectronProducers
  edm::Handle<reco::ElectronCollection> electronIsolatedHandle;
  iEvent.getByToken(electronIsolatedToken_,electronIsolatedHandle);

  edm::Handle<reco::ElectronCollection> electronNonIsolatedHandle;
  if(!doIsolated_) {
    iEvent.getByToken(electronNonIsolatedToken_,electronNonIsolatedHandle);
  }

 // look at all candidates,  check cuts and add to filter object
  int n(0);

    //loop over all the RecoCandidates from the previous filter,
    // associate them with the corresponding Electron object
    //(the matching is done checking the super clusters)
    // and put into the event a Ref to the Electron objects that passes the
    // selections
  for (unsigned int i=0; i<recoecalcands.size(); i++) {
    reco::SuperClusterRef recr2 = recoecalcands[i]->superCluster();

    //loop over the electrons to find the matching one
    for(reco::ElectronCollection::const_iterator iElectron = electronIsolatedHandle->begin(); iElectron != electronIsolatedHandle->end(); iElectron++){

      reco::ElectronRef electronref(reco::ElectronRef(electronIsolatedHandle,iElectron - electronIsolatedHandle->begin()));
      const reco::SuperClusterRef theClus = electronref->superCluster();

      if(&(*recr2) ==  &(*theClus)) {

	float elecEoverp = 0;
	const math::XYZVector trackMom =  electronref->track()->momentum();
	if( trackMom.R() != 0) elecEoverp =
				 electronref->superCluster()->energy()/ trackMom.R();

	if( fabs(electronref->eta()) < 1.5 ){
	  if ( elecEoverp < eoverpbarrelcut_) {
	    n++;
	    filterproduct.addObject(TriggerElectron, electronref);
	  }
	}
	if( fabs(electronref->eta()) > 1.5 ){
	  if ( elecEoverp < eoverpendcapcut_) {
	    n++;
	    filterproduct.addObject(TriggerElectron, electronref);
	  }
	}
      }//end of the if checking the matching of the SC from RecoCandidate and the one from Electrons
    }//end of loop over electrons

    if(!doIsolated_) {
    //loop over the electrons to find the matching one
    for(reco::ElectronCollection::const_iterator iElectron = electronNonIsolatedHandle->begin(); iElectron != electronNonIsolatedHandle->end(); iElectron++){

      reco::ElectronRef electronref(reco::ElectronRef(electronNonIsolatedHandle,iElectron - electronNonIsolatedHandle->begin()));
      const reco::SuperClusterRef theClus = electronref->superCluster();

      if(&(*recr2) ==  &(*theClus)) {

	float elecEoverp = 0;
	const math::XYZVector trackMom =  electronref->track()->momentum();
	if( trackMom.R() != 0) elecEoverp =
				 electronref->superCluster()->energy()/ trackMom.R();

	if( fabs(electronref->eta()) < 1.5 ){
	  if ( elecEoverp < eoverpbarrelcut_) {
	    n++;
	    filterproduct.addObject(TriggerElectron, electronref);
	  }
	}
	if( fabs(electronref->eta()) > 1.5 ){
	  if ( elecEoverp < eoverpendcapcut_) {
	    n++;
	    filterproduct.addObject(TriggerElectron, electronref);
	  }
	}
      }//end of the if checking the matching of the SC from RecoCandidate and the one from Electrons
    }//end of loop over electrons
    }
  }//end of loop ober candidates

  // filter decision
  bool accept(n>=ncandcut_);

  return accept;
}
