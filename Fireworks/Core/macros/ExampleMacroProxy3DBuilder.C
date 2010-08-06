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
// $Id: ExampleMacroProxy3DBuilder.C,v 1.4 2009/01/23 21:35:42 amraktad Exp $
//

// system include files
#include "TEveManager.h"
#include "TEveTrack.h"
#include "TEveVSDStructs.h"

class FWDataProxyBuilder;

#if !defined(__CINT__) && !defined(__MAKECINT__)
// user include files
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWRPZDataProxyBuilder.h"
#include "TEveVSDStructs.h"

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
      TEveElementList* tlist = *product;
      if(!tlist) {
         tlist =  new TEveElementList(iItem->name().c_str());
         *product = tlist;
      }

      const reco::TrackCollection* tracks=0;
      iItem->get(tracks);
      if(0 == tracks ) return;


      int index=0;
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

         TEveTrack* trk = new TEveTrack(&t,context().getTrackPropagator());
         trk->SetMainColor(iItem->displayProperties().color());
         trk->MakeTrack();
         tlist->AddElement(trk);
      }
   }


   ExampleMacroProxy3DBuilder(const ExampleMacroProxy3DBuilder &);   // stop default

   const ExampleMacroProxy3DBuilder& operator=(const ExampleMacroProxy3DBuilder &);   // stop default

   // ---------- member data --------------------------------

};


