/** \class HLTDeDxFilter
*
*
*  \author Claude Nuttens
*
*/

#include "RecoTracker/DeDx/plugins/HLTDeDxFilter.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/TrackReco/interface/DeDxData.h"
//#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include <vector>


//
// constructors and destructor
//
HLTDeDxFilter::HLTDeDxFilter(const edm::ParameterSet& iConfig)
{
  saveTags_     = iConfig.getParameter<bool>("saveTags");
  minDEDx_	= iConfig.getParameter<double> ("minDEDx");
  minPT_        = iConfig.getParameter<double> ("minPT");
  minNOM_       = iConfig.getParameter<double> ("minNOM");
  maxETA_       = iConfig.getParameter<double> ("maxETA");
  inputTracksTag_ = iConfig.getParameter< edm::InputTag > ("inputTracksTag");
  inputdedxTag_   = iConfig.getParameter< edm::InputTag > ("inputDeDxTag");
 
  //register your products
  produces<trigger::TriggerFilterObjectWithRefs>();
}

HLTDeDxFilter::~HLTDeDxFilter(){}

void HLTDeDxFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<bool>("saveTags",false);
  desc.add<double>("minDEDx",0.0);
  desc.add<double>("minPT",0.0);
  desc.add<double>("minNOM",0.0);
  desc.add<double>("maxETA",5.5);
  desc.add<edm::InputTag>("inputTracksTag",edm::InputTag("hltL3Mouns"));
  desc.add<edm::InputTag>("inputDeDxTag",edm::InputTag("HLTdedxHarm2"));
  descriptions.add("hltDeDxFilter",desc);
}



// ------------ method called to produce the data  ------------
bool
  HLTDeDxFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace std;
  using namespace edm;
  using namespace reco;
  using namespace trigger;
  // The filter object
  auto_ptr<trigger::TriggerFilterObjectWithRefs> filterobject (new trigger::TriggerFilterObjectWithRefs(path(),module()));
  //  if (saveTags_) filterobject->addCollectionTag(inputJetTag_);

   edm::Handle<reco::TrackCollection> trackCollectionHandle;
   iEvent.getByLabel(inputTracksTag_,trackCollectionHandle);
   reco::TrackCollection trackCollection = *trackCollectionHandle.product();
  
   edm::Handle<edm::ValueMap<reco::DeDxData> > dEdxTrackHandle;
   //iEvent.getByLabel(m_dEdxDiscrimTag, dEdxTrackHandle);
   //iEvent.getByLabel("dedxHarmonic2", dEdxTrackHandle);
   iEvent.getByLabel(inputdedxTag_, dEdxTrackHandle);
   const edm::ValueMap<reco::DeDxData> dEdxTrack = *dEdxTrackHandle.product();

   bool accept=false;
   for(unsigned int i=0; i<trackCollection.size(); i++){
      reco::TrackRef track  = reco::TrackRef( trackCollectionHandle, i );
     //Track momentum is given by:
     //track->p();
     //You can access dE/dx Estimation of your track with:
     if(track->pt()>minPT_ && fabs(track->eta())<maxETA_ && dEdxTrack[track].numberOfMeasurements()>minNOM_ && dEdxTrack[track].dEdx()>minDEDx_){accept=true; break;};
   }
   // put filter object into the Event
   iEvent.put(filterobject);
   return accept;
}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTDeDxFilter);
