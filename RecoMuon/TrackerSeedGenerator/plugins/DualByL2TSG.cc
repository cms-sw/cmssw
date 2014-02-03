#include "RecoMuon/TrackerSeedGenerator/plugins/DualByL2TSG.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/MuonSeed/interface/L3MuonTrajectorySeed.h"
#include "DataFormats/MuonSeed/interface/L3MuonTrajectorySeedCollection.h"
#include "RecoMuon/TrackerSeedGenerator/interface/TrackerSeedGenerator.h"

DualByL2TSG::DualByL2TSG(const edm::ParameterSet &pset) : SeparatingTSG(pset){  theCategory ="DualByL2TSG";
  theL3CollectionLabelA = pset.getParameter<edm::InputTag>("L3TkCollectionA");
  if (nTSGs()!=2)
    {edm::LogError(theCategory)<<"not two seed generators provided";}
}

unsigned int DualByL2TSG::selectTSG(const TrackCand & muonTrackCand, const TrackingRegion& region)
{
  LogDebug(theCategory)<<"|eta|=|"<<muonTrackCand.second->eta()<<"|";

  bool re_do_this_L2 = true;
  //LogDebug("TrackerSeedGenerator")<<theEvent;
  //getEvent();
  
  //retrieve L3 track collection
  edm::Handle<reco::TrackCollection> l3muonH;
  getEvent()->getByLabel(theL3CollectionLabelA ,l3muonH);
  if(l3muonH.failedToGet()) return 0;
  
  unsigned int maxI = l3muonH->size();
  
  LogDebug(theCategory) << "TheCollectionA size " << maxI;

  // Loop through all tracks, if the track was seeded from this L2, then skip
  for (unsigned int i=0;i!=maxI;++i){
    reco::TrackRef tk(l3muonH,i);
    edm::Ref<L3MuonTrajectorySeedCollection> l3seedRef = tk->seedRef().castTo<edm::Ref<L3MuonTrajectorySeedCollection> >();
    reco::TrackRef staTrack = l3seedRef->l2Track();

    if(staTrack == (muonTrackCand.second) ) re_do_this_L2 = false;
    //LogDebug(theCategory) << "The DualByL2TSG selectTSG loop " << re_do_this_L2 << " staCand " << muonTrackCand.second->eta() << " " << muonTrackCand.second->pt() << " alreadyMadeRefToL3 " << staTrack->eta() << " " << staTrack->pt();
  }
  
  LogDebug(theCategory) << "The DualByL2TSG to use " << re_do_this_L2 ;

  return re_do_this_L2 ? 1 : 0;
}
