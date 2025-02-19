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
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"

//
// constructors and destructor
//
HLTDeDxFilter::HLTDeDxFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig)
{
  minDEDx_	= iConfig.getParameter<double> ("minDEDx");
  minPT_        = iConfig.getParameter<double> ("minPT");
  minNOM_       = iConfig.getParameter<double> ("minNOM");
  maxETA_       = iConfig.getParameter<double> ("maxETA");
  inputTracksTag_ = iConfig.getParameter< edm::InputTag > ("inputTracksTag");
  inputdedxTag_   = iConfig.getParameter< edm::InputTag > ("inputDeDxTag");

  thisModuleTag_ = edm::InputTag(iConfig.getParameter<std::string>("@module_label")); 
 
  //register your products
  produces<reco::RecoChargedCandidateCollection>();
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
  HLTDeDxFilter::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct)
{
  using namespace std;
  using namespace edm;
  using namespace reco;
  using namespace trigger;

  auto_ptr<RecoChargedCandidateCollection> chargedCandidates( new std::vector<RecoChargedCandidate> );

  ModuleDescription moduleDesc_;

  if (saveTags()){
     filterproduct.addCollectionTag(thisModuleTag_);
     filterproduct.addCollectionTag(inputTracksTag_);
     filterproduct.addCollectionTag(inputdedxTag_);
  }

  edm::Handle<reco::TrackCollection> trackCollectionHandle;
  iEvent.getByLabel(inputTracksTag_,trackCollectionHandle);
  reco::TrackCollection trackCollection = *trackCollectionHandle.product();
  
  edm::Handle<edm::ValueMap<reco::DeDxData> > dEdxTrackHandle;
  iEvent.getByLabel(inputdedxTag_, dEdxTrackHandle);
  const edm::ValueMap<reco::DeDxData> dEdxTrack = *dEdxTrackHandle.product();

  bool accept=false;
  int  NTracks = 0;
  for(unsigned int i=0; i<trackCollection.size(); i++){
     reco::TrackRef track  = reco::TrackRef( trackCollectionHandle, i );
    if(track->pt()>minPT_ && fabs(track->eta())<maxETA_ && dEdxTrack[track].numberOfMeasurements()>minNOM_ && dEdxTrack[track].dEdx()>minDEDx_){
       NTracks++;
       if (saveTags()){
          Particle::Charge q = track->charge();
          //SAVE DEDX INFORMATION AS IF IT WAS THE MASS OF THE PARTICLE
          Particle::LorentzVector p4(track->px(), track->py(), track->pz(), sqrt(pow(track->p(),2) + pow(dEdxTrack[track].dEdx(),2)));
          Particle::Point vtx(track->vx(),track->vy(), track->vz());
          //SAVE NOH, NOM, NOS INFORMATION AS IF IT WAS THE PDGID OF THE PARTICLE
          int Hits  = ((dEdxTrack[track].numberOfSaturatedMeasurements()&0xFF)<<16) | ((dEdxTrack[track].numberOfMeasurements()&0xFF)<<8) | (track->found()&0xFF); 
          RecoChargedCandidate cand(q, p4, vtx, Hits, 0);
          cand.setTrack(track);
          chargedCandidates->push_back(cand);
       }
       accept=true; 
    }
  }

  // put filter object into the Event
   if(saveTags()){
     edm::OrphanHandle<RecoChargedCandidateCollection> chargedCandidatesHandle = iEvent.put(chargedCandidates);
     for(int i=0; i<NTracks; i++){
          filterproduct.addObject(TriggerMuon,RecoChargedCandidateRef(chargedCandidatesHandle,i));
     }
   }

  return accept;
}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTDeDxFilter);
