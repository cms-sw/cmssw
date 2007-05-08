#ifndef JetProducers_JetTrackAssociator_h
#define JetProducers_JetTrackAssociator_h

/// Abstract interface to fill JetTrack association
/// \author: F.Ratnikov, UMd
/// Apr. 20, 2007
/// $Id: JetTrackAssociator.h,v 1.1 2007/05/03 21:20:09 fedor Exp $

#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/JetReco/interface/JetTrackMatch.h"


template <typename JetC>
class JetTrackAssociator {
 public:
  typedef edm::Ref<JetC> JetRef;
  typedef reco::TrackCollection TrackC;
  typedef edm::Ref<TrackC> TrackRef;
  typedef reco::JetTrackMatch<JetC> MatchMap;
  JetTrackAssociator () {}
  virtual ~JetTrackAssociator () {}

  /// virtual method to make association: to be defined in concrete class
  virtual bool associate (const JetRef& fJet, const TrackRef& fTrack) const = 0;

  /// selection criteria for "good" track
  virtual bool goodTrack (const TrackRef& fTrack) const {return true;}

  /// association engine
  void buildMap (const edm::Handle<JetC>& fJets, const edm::Handle<TrackC>& fTracks, MatchMap* fMap) const {
    if (!fMap) {
      throw cms::Exception ("JetTrackAssociator") << " Invalid supplied matching map pointer" << std::endl;
    }
    for (unsigned iJet = 0; iJet < fJets->size(); ++iJet) {
      typename MatchMap::JetRef jetRef (fJets, iJet);  
      bool orphanJet = true;
      for (unsigned iTrack = 0; iTrack < fTracks->size(); ++iTrack) {
	typename MatchMap::TrackRef trackRef (fTracks, iTrack); 
	if (goodTrack (trackRef) && associate (jetRef, trackRef)) {
	  fMap->insert (jetRef, trackRef);
	  orphanJet = false;
	}
      }
      if (orphanJet) fMap->insert (jetRef);
    }
  }
};

#endif
