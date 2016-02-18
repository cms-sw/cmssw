#include "FastSimulation/Tracking/plugins/ElectronSeedTrackRefFix.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/ParticleFlowReco/interface/PreId.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/Common/interface/ValueMap.h"
//
// class declaration
//




//
// constructors and destructor
//

using namespace std;
using namespace reco;
using namespace edm;

ElectronSeedTrackRefFix::ElectronSeedTrackRefFix(const edm::ParameterSet& iConfig)
{
  // read parameters
  preidgsfLabel = iConfig.getParameter<string>("PreGsfLabel");
  preidLabel= iConfig.getParameter<string>("PreIdLabel");
  oldTracksTag = iConfig.getParameter<InputTag>("oldTrackCollection");
  newTracksTag = iConfig.getParameter<InputTag>("newTrackCollection");
  seedsTag = iConfig.getParameter<InputTag>("seedCollection");
  idsTag = iConfig.getParameter<InputTag>("idCollection");
  
  //register your products
  produces<reco::ElectronSeedCollection>(preidgsfLabel);
  produces<reco::PreIdCollection>(preidLabel);
  produces<ValueMap<reco::PreIdRef> >(preidLabel);

  //create tokens
  oldTracksToken = consumes<reco::TrackCollection>(oldTracksTag);
  newTracksToken = consumes<reco::TrackCollection>(newTracksTag);
  seedsToken = consumes<reco::ElectronSeedCollection>(seedsTag);
  idsToken = consumes<reco::PreIdCollection >(idsTag);
  idMapToken = consumes<ValueMap<reco::PreIdRef> >(idsTag) ;
}


ElectronSeedTrackRefFix::~ElectronSeedTrackRefFix()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
ElectronSeedTrackRefFix::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;

   Handle<TrackCollection> oldTracks;
   iEvent.getByToken(oldTracksToken,oldTracks);

   Handle<TrackCollection> newTracks;
   iEvent.getByToken(newTracksToken,newTracks);
   
   Handle<ElectronSeedCollection> iSeeds;
   iEvent.getByToken(seedsToken,iSeeds);
   
   Handle<PreIdCollection > iIds;
   iEvent.getByToken(idsToken,iIds);
   
   Handle<ValueMap<PreIdRef> > iIdMap;
   iEvent.getByToken(idMapToken,iIdMap);

  auto_ptr<ElectronSeedCollection> oSeeds(new ElectronSeedCollection);
  auto_ptr<PreIdCollection> oIds(new PreIdCollection);
  auto_ptr<ValueMap<PreIdRef> > oIdMap(new ValueMap<PreIdRef>);

   ValueMap<PreIdRef>::Filler mapFiller(*oIdMap);
   
   for(unsigned int s = 0;s<iSeeds->size();++s){
     oSeeds->push_back(iSeeds->at(s));
     TrackRef newTrackRef(newTracks,oSeeds->back().ctfTrack().index());
     oSeeds->back().setCtfTrack(newTrackRef);
   }

   for(unsigned int i = 0;i<iIds->size();++i){
     oIds->push_back(iIds->at(i));
     TrackRef newTrackRef(newTracks,oIds->back().trackRef().index());
     oIds->back().setTrack(newTrackRef);
   }

   iEvent.put(oSeeds,preidgsfLabel);
   const edm::OrphanHandle<reco::PreIdCollection> preIdProd = iEvent.put(oIds,preidLabel);

   vector<PreIdRef> values;
   for(unsigned int t = 0;t<newTracks->size();++t){
     if(t < oldTracks->size()){
       TrackRef oldTrackRef(oldTracks,t);
       values.push_back(PreIdRef(preIdProd,(*(iIdMap.product()))[oldTrackRef].index()));
     }
     else{
       values.push_back(PreIdRef());
     }
   }
   mapFiller.insert(newTracks,values.begin(),values.end());
   mapFiller.fill();

   iEvent.put(oIdMap,preidLabel);
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
ElectronSeedTrackRefFix::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

