#ifndef JetReco_JetTrackMatch_h
#define JetReco_JetTrackMatch_h

/** \class reco::JetTrackMatch
 *
 * \short Association between Jets from jet collection and tracks from track collection
 *
 * Every jet may have several tracks associated with it
 *
 * class definition:
 * 
 *  template <typename JetC>
 *   class JetTrackMatch {
 *   public:
 *   typedef edm::Ref<JetC> JetRef;
 *   typedef edm::Ref<reco::TrackCollection> TrackRef;
 *   JetTrackMatch ();
 *   ~JetTrackMatch ();
 *   // insert orphan jet
 *   void insert (const JetRef& fJet);
 *   // assign track to jet. 
 *   void insert (const JetRef& fJet, const TrackRef& fTrack);
 *   // get list of all tracks in the map
 *   std::vector <JetRef> allJets () const;
 *   // get all tracks associated with jet
 *   std::vector <TrackRef> getTracks (const JetRef& mJet) const;
 * };
 *
 * \author Fedor Ratnikov, UMd
 *
 * \version   $Id: JetTrackMatch.h,v 1.2 2007/09/18 13:35:24 ratnik Exp $
 ************************************************************/

#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

namespace reco {
  template <typename JetC>
    class JetTrackMatch {
    public:
    typedef edm::Ref<JetC> JetRef;
    typedef edm::Ref<reco::TrackCollection> TrackRef;
    typedef edm::AssociationMap<edm::OneToMany<JetC, reco::TrackCollection> > Map;
    private:
    Map mMap;

    public:
    JetTrackMatch () {}
    ~JetTrackMatch () {}

    /// insert orphan jet
    void insert (const JetRef& fJet) {
      mMap.insert (fJet, TrackRef());
    }

    /// assign track to jet. 
    void insert (const JetRef& fJet, const TrackRef& fTrack) {
      mMap.insert (fJet, fTrack);
    }

    /// get list of all jats in the map
    std::vector <JetRef> allJets () const {
      std::vector <JetRef> result;
      typename Map::const_iterator it = mMap.begin ();
      for (; it != mMap.end(); ++it) {
	result.push_back (it->key);
      }
      return result;
    }
    /// get all tracks associated with jet
    std::vector <TrackRef> getTracks (const JetRef& mJet) const {
      std::vector <TrackRef> result;
      reco::TrackRefVector tracks = mMap [mJet];
      int i = tracks.size();
      while (--i >= 0) {
	if (!tracks [i].isNull ()) result.push_back (tracks [i]);
      }
      return result;
    }
  };
}

#endif
