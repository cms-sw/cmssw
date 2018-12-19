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






namespace{
  template<typename VMType,typename HandleType>
  const edm::OrphanHandle<edm::ValueMap<VMType> > addVMToEvent(edm::Event& event,const edm::Handle<HandleType>& handle,std::vector<VMType> values,const std::string& label)
  { 
    auto vMap = std::make_unique<edm::ValueMap<VMType> >();
    typename edm::ValueMap<VMType>::Filler mapFiller(*vMap);
    mapFiller.insert(handle,values.begin(),values.end());
    mapFiller.fill();
    return event.put(std::move(vMap),label);
  }
  template<typename T> edm::Handle<T> getHandle(const edm::Event& event,const edm::EDGetTokenT<T>& token)
  {
    edm::Handle<T> handle;
    event.getByToken(token,handle);
    return handle;
  }
}

ElectronSeedTrackRefFix::ElectronSeedTrackRefFix(const edm::ParameterSet& iConfig)
{
  // read parameters
  preidgsfLabel_ = iConfig.getParameter<std::string>("PreGsfLabel");
  preidLabel_ = iConfig.getParameter<std::vector<std::string> >("PreIdLabel");
  oldTracksTag_ = iConfig.getParameter<edm::InputTag>("oldTrackCollection");
  newTracksTag_ = iConfig.getParameter<edm::InputTag>("newTrackCollection");
  seedsTag_ = iConfig.getParameter<edm::InputTag>("seedCollection");
  idsTag_ = iConfig.getParameter<std::vector<edm::InputTag>>("idCollection");
  
  //register your products
  produces<reco::ElectronSeedCollection>(preidgsfLabel_);
  for(const auto& idLabel : preidLabel_){
    produces<reco::PreIdCollection>(idLabel);
    produces<edm::ValueMap<reco::PreIdRef> >(idLabel);
  }

  //create tokens
  oldTracksToken_ = consumes<reco::TrackCollection>(oldTracksTag_);
  newTracksToken_ = consumes<reco::TrackCollection>(newTracksTag_);
  seedsToken_ = consumes<reco::ElectronSeedCollection>(seedsTag_);

  for(const auto& idTag: idsTag_){
    idsToken_.emplace_back(consumes<reco::PreIdCollection >(idTag));
    idMapToken_.emplace_back(consumes<edm::ValueMap<reco::PreIdRef> >(idTag));
  }
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

   auto oldTracks = getHandle(iEvent,oldTracksToken_);
   auto newTracks = getHandle(iEvent,newTracksToken_);
   auto iSeeds = getHandle(iEvent,seedsToken_);
   
   auto oSeeds = std::make_unique<reco::ElectronSeedCollection>(); 
   for(unsigned int s = 0;s<iSeeds->size();++s){
     oSeeds->push_back(iSeeds->at(s));
     reco::TrackRef newTrackRef(newTracks,oSeeds->back().ctfTrack().index());
     oSeeds->back().setCtfTrack(newTrackRef);
   }
   iEvent.put(std::move(oSeeds),preidgsfLabel_);

   for(size_t idNr = 0; idNr < idsTag_.size(); idNr++){
     auto iIds = getHandle(iEvent,idsToken_[idNr]);
    
     auto oIds = std::make_unique<reco::PreIdCollection>();
     for(unsigned int i = 0;i<iIds->size();++i){
       oIds->push_back(iIds->at(i));
       reco::TrackRef newTrackRef(newTracks,oIds->back().trackRef().index());
       oIds->back().setTrack(newTrackRef);
     }
     const edm::OrphanHandle<reco::PreIdCollection> preIdProd = iEvent.put(std::move(oIds),preidLabel_[idNr]);

     auto iIdMap = getHandle(iEvent,idMapToken_[idNr]);
     std::vector<reco::PreIdRef> values;
     for(size_t trkNr = 0;trkNr<newTracks->size();++trkNr){
       //low pt electron seeds do not make the idMaps so this is now optional to fill in a useful way
       if(trkNr < oldTracks->size() && iIdMap.isValid()){
	 reco::TrackRef oldTrackRef(oldTracks,trkNr);
	 values.push_back(reco::PreIdRef(preIdProd,(*(iIdMap.product()))[oldTrackRef].index()));
       }else values.push_back(reco::PreIdRef());
     }
     addVMToEvent(iEvent,newTracks,std::move(values),preidLabel_[idNr]);
   }//end loop over ids
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

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(ElectronSeedTrackRefFix);
