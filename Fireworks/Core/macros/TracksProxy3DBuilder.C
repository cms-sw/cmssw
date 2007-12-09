// -*- C++ -*-
//
// Package:     Core
// Class  :     TracksProxy3DBuilder
// 
/**\class TracksProxy3DBuilder TracksProxy3DBuilder.h Fireworks/Core/interface/TracksProxy3DBuilder.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  
//         Created:  Thu Dec  6 18:01:21 PST 2007
// $Id$
//

// system include files
#include "TEveManager.h"
#include "TEveTrack.h"
#include "TEveTrackPropagator.h"
#include "DataFormats/FWLite/interface/Event.h"
#include "DataFormats/FWLite/interface/Handle.h"

namespace fwlite {
  class Event;
}
class FWDataProxyBuilder;

#if !defined(__CINT__) && !defined(__MAKECINT__)
// user include files
#include "Fireworks/Core/interface/FWDataProxyBuilder.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#endif

// forward declarations

class TracksProxy3DBuilder : public FWDataProxyBuilder
{

   public:
      TracksProxy3DBuilder() {}
      virtual ~TracksProxy3DBuilder() {}

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      virtual void build(const fwlite::Event* iEvent,
			 TEveElementList** oList)
  {
    std::cout <<"build called"<<std::endl;
    fwlite::Handle<reco::TrackCollection> tracks;
    tracks.getByLabel(*iEvent,"ctfWithMaterialTracks");
    
    if(0 == tracks.ptr() ) {
      std::cout <<"failed to get Tracks"<<std::endl;
    }
    
    if(0 == *oList) {
      TEveTrackList* tlist =  new TEveTrackList("Tracks");
      *oList =tlist;
      (*oList)->SetMainColor(Color_t(3));
      TEveTrackPropagator* rnrStyle = tlist->GetPropagator();
      //units are kG
      rnrStyle->SetMagField( -4.0*10.);
      //get this from geometry, units are CM
      rnrStyle->SetMaxR(120.0);
      rnrStyle->SetMaxZ(300.0);
      
      gEve->AddElement(*oList);
    } else {
      (*oList)->DestroyElements();
    }
    //since we created it, we know the type (would like to do this better)
    TEveTrackList* tlist = dynamic_cast<TEveTrackList*>(*oList);
    
    TEveTrackPropagator* rnrStyle = tlist->GetPropagator();
    
    int index=0;
    //cout <<"----"<<endl;
    TEveRecTrack t;
    t.beta = 1.;
    for(reco::TrackCollection::const_iterator it = tracks->begin();
	it != tracks->end();++it,++index) {
      t.P = TEveVector(it->px(),
		       it->py(),
		       it->pz());
      t.V = TEveVector(it->vx(),
		       it->vy(),
		       it->vz());
      t.sign = it->charge();
      
      TEveTrack* trk = new TEveTrack(&t,rnrStyle);
      trk->SetMainColor((*oList)->GetMainColor());
      gEve->AddElement(trk,(*oList));
      //cout << it->px()<<" "
      //   <<it->py()<<" "
      //   <<it->pz()<<endl;
      //cout <<" *";
    }
    
  }

   private:
      TracksProxy3DBuilder(const TracksProxy3DBuilder&); // stop default

      const TracksProxy3DBuilder& operator=(const TracksProxy3DBuilder&); // stop default

      // ---------- member data --------------------------------

};


