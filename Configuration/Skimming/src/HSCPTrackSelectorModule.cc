
// -*- C++ -*-
//
// Package:    HSCPTrackSelector
// Class:      HSCPTrackSelector
// 
// Original Author:  Loic Quertenmont
//         Created:  Mon Apr 23 CDT 2012
//


#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/ObjectSelector.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include <DataFormats/MuonReco/interface/Muon.h>
#include <DataFormats/MuonReco/interface/MuonFwd.h>
#include "DataFormats/TrackReco/interface/DeDxData.h"
#include "DataFormats/Common/interface/ValueMap.h"

// the following include is necessary to clone all track branches
// including recoTrackExtras and TrackingRecHitsOwned (in future also "owned clusters"?).
// if you remove it the code will compile, but the cloned
// tracks have only the recoTracks branch!
#include "CommonTools/RecoAlgos/interface/TrackSelector.h"

struct HSCPTrackSelector {
  typedef std::vector<const reco::Track*> container;
  typedef container::const_iterator const_iterator;
  typedef reco::TrackCollection collection; 

 HSCPTrackSelector( const edm::ParameterSet & cfg ){
    trackerTrackPtCut = cfg.getParameter<double>( "trackerTrackPtMin" );
    dedxTag_          = cfg.getParameter<edm::InputTag>("InputDedx");
    theMuonSource     = cfg.getParameter<edm::InputTag>("muonSource");

    thedEdxSwitch         = cfg.getParameter<bool>("usededx");
    minInnerTrackdEdx     = cfg.getParameter<double>( "InnerTrackdEdxRightMin" );
    maxInnerTrackdEdx     = cfg.getParameter<double>( "InnerTrackdEdxLeftMax" );
    minMuonTrackdEdx      = cfg.getParameter<double>( "InnerMuondEdxRightMin" );
    maxMuonTrackdEdx      = cfg.getParameter<double>( "InnerMuondEdxLeftMax" );
    mindEdxHitsInnerTrack = cfg.getParameter<unsigned int>( "dEdxMeasurementsMinForMuonTrack" );
    mindEdxHitsMuonTrack  = cfg.getParameter<unsigned int>( "dEdxMeasurementsMinForInnerTrack" );
  }
  
  const_iterator begin() const { return theSelectedTracks.begin(); }
  const_iterator end() const { return theSelectedTracks.end(); }
  size_t size() const { return theSelectedTracks.size(); }

  bool matchingMuon(const reco::Track* track, const edm::Handle<reco::MuonCollection>& muons){
    for(reco::MuonCollection::const_iterator itMuon = muons->begin(); itMuon != muons->end(); ++itMuon){
      const reco::Track* muonTrack = (*itMuon).get<reco::TrackRef>().get();
      if (!muonTrack)continue;

      //matching is needed because input track collection (made for dE/dx) has been refitted w.r.t track collection used for muon reco (we should do a tight maching to find equivalent track in the other collection)
      if(fabs(track->pt()-muonTrack->pt())<0.5 && fabs(track->eta()-muonTrack->eta())<0.02 && fabs(track->phi()-muonTrack->phi())<0.02)return true;
    }
    return false;
  }



  void select( const edm::Handle<reco::TrackCollection> & c,  const edm::Event & evt,  const edm::EventSetup &){
    edm::Handle<edm::ValueMap<reco::DeDxData> > dEdxTrackHandle;
    if(thedEdxSwitch){evt.getByLabel(dedxTag_, dEdxTrackHandle); }

    edm::Handle<reco::MuonCollection> muons;
    evt.getByLabel(theMuonSource, muons);

    //Loop on all Tracks
    theSelectedTracks.clear();
    for(size_t i=0; i<c->size(); i++){
      reco::TrackRef trkRef = reco::TrackRef(c, i);
      
      double dedx=0; unsigned int dedxHit=0; if(thedEdxSwitch){dedx=dEdxTrackHandle->get(i).dEdx(); dedxHit=dEdxTrackHandle->get(i).numberOfMeasurements();}  //DIRTY WAY OF ACCESSING DEDX
      bool isMuon=muons.isValid() && matchingMuon(& *trkRef, muons);
      //printf("muon tracks %i --> pt=%6.2f  eta=%+6.2f phi=%+6.2f dEdx=%f\n", (int)isMuon,  trkRef->pt(), trkRef->eta(), trkRef->phi(), dedx);
      
      if(isMuon){
         if(thedEdxSwitch && (dedxHit<mindEdxHitsMuonTrack || (dedx>minMuonTrackdEdx && dedx<maxMuonTrackdEdx)) ){continue;}     
         theSelectedTracks.push_back(& * trkRef );
      }else{
         if(trkRef->pt()<trackerTrackPtCut)continue;
         if(thedEdxSwitch && (dedxHit<mindEdxHitsInnerTrack || (dedx>minInnerTrackdEdx && dedx<maxInnerTrackdEdx)) )continue;
         theSelectedTracks.push_back(& * trkRef );
      }
    }

    //debug printout
//    for (container::const_iterator it=theSelectedTracks.begin(); it != theSelectedTracks.end(); ++it) {
//       printf("Selected tracks %i --> pt=%6.2f  eta=%+6.2f phi=%+6.2f - isMuon=%i\n", (int)(it-theSelectedTracks.begin()),  (*it)->pt(), (*it)->eta(), (*it)->phi(), matchingMuon(*it, theSelectedMuonTracks));
//    }
  }

private:
  container theSelectedTracks;
  double trackerTrackPtCut;
  double minInnerTrackdEdx;
  double maxInnerTrackdEdx;
  double minMuonTrackdEdx;
  double maxMuonTrackdEdx;
  unsigned int    mindEdxHitsInnerTrack;
  unsigned int    mindEdxHitsMuonTrack;
  edm::InputTag dedxTag_;
  edm::InputTag theMuonSource;
  bool thedEdxSwitch;
};

typedef ObjectSelector<HSCPTrackSelector>  HSCPTrackSelectorModule;

DEFINE_FWK_MODULE( HSCPTrackSelectorModule );
