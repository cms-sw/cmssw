// -*- C++ -*-
//
// Package:     Core
// Class  :     ExampleMacroProxy3DBuilder
//
/**\class ExampleMacroProxy3DBuilder ExampleMacroProxy3DBuilder.h Fireworks/Core/interface/ExampleMacroProxy3DBuilder.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:
//         Created:  Thu Dec  6 18:01:21 PST 2007
// $Id: ExampleMacroProxy3DBuilder.C,v 1.3 2008/07/20 18:22:00 dmytro Exp $
//

// system include files
#include "TEveManager.h"
#include "TEveTrack.h"
#include "TEveTrackPropagator.h"

class FWDataProxyBuilder;

#if !defined(__CINT__) && !defined(__MAKECINT__)
// user include files
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWRPZDataProxyBuilder.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#endif

// forward declarations
class FWRPZDataProxyBuilder;

class ExampleMacroProxy3DBuilder : public FWRPZDataProxyBuilder
{

public:
   ExampleMacroProxy3DBuilder() {
   }
   virtual ~ExampleMacroProxy3DBuilder() {}

   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
private:
   virtual void build(const FWEventItem* iItem,
                      TEveElementList** product)
   {
      //since we created it, we know the type (would like to do this better)
      TEveTrackList* tlist = dynamic_cast<TEveTrackList*>(*product);
      if ( !tlist && *product ) {
         std::cout << "incorrect type" << std::endl;
         return;
      }

      if(0 == tlist) {
         tlist =  new TEveTrackList(iItem->name().c_str());
         *product = tlist;
         tlist->SetMainColor(iItem->displayProperties().color());
         TEveTrackPropagator* rnrStyle = tlist->GetPropagator();
         //units are kG
         rnrStyle->SetMagField( -4.0*10.);
         //get this from geometry, units are CM
         rnrStyle->SetMaxR(120.0);
         rnrStyle->SetMaxZ(300.0);

         gEve->AddElement(tlist);
      } else {
         tlist->DestroyElements();
      }

      const reco::TrackCollection* tracks=0;
      iItem->get(tracks);
      //fwlite::Handle<reco::TrackCollection> tracks;
      //tracks.getByLabel(*iEvent,"ctfWithMaterialTracks");

      if(0 == tracks ) return;

      TEveTrackPropagator* rnrStyle = tlist->GetPropagator();

      int index=0;
      //cout <<"----"<<endl;
      TEveRecTrack t;
      t.beta = 1.;
      for(reco::TrackCollection::const_iterator it = tracks->begin();
          it != tracks->end(); ++it,++index) {
         t.P = TEveVector(it->px(),
                          it->py(),
                          it->pz());
         t.V = TEveVector(it->vx(),
                          it->vy(),
                          it->vz());
         t.sign = it->charge();

         TEveTrack* trk = new TEveTrack(&t,rnrStyle);
         trk->SetMainColor(iItem->displayProperties().color());
         gEve->AddElement(trk,tlist);
         //cout << it->px()<<" "
         //   <<it->py()<<" "
         //   <<it->pz()<<endl;
         //cout <<" *";
      }

   }


   ExampleMacroProxy3DBuilder(const ExampleMacroProxy3DBuilder &);   // stop default

   const ExampleMacroProxy3DBuilder& operator=(const ExampleMacroProxy3DBuilder &);   // stop default

   // ---------- member data --------------------------------

};


