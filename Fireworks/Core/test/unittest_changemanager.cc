// -*- C++ -*-
//
// Package:     Core
// Class  :     unittest_changemanager
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Fri Jan 18 10:19:07 EST 2008
// $Id: unittest_changemanager.cc,v 1.4 2008/03/05 16:43:12 chrjones Exp $
//

// system include files
#include <boost/test/auto_unit_test.hpp>
#include <boost/bind.hpp>
#include <boost/test/test_tools.hpp>

#include "TClass.h"
#include "Cintex/Cintex.h"

// user include files
#include "Fireworks/Core/interface/FWModelChangeManager.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"


//
// constants, enums and typedefs
//
namespace {
   struct Listener {
      Listener(): nHeard_(0) {}
      int nHeard_;
      
      void listen(const std::set<FWModelId>& iIds) {
         nHeard_  += iIds.size();
      }
   };
   
   struct ItemListener {
      ItemListener(): nHeard_(0){}
      int nHeard_;
      
      void listen(const FWEventItem* iItem) {
         ++nHeard_;
      }
   };
}

BOOST_AUTO_TEST_CASE( changemanager )
{
   ROOT::Cintex::Cintex::Enable();

   FWModelChangeManager cm;
   
   //create an item
   reco::TrackCollection fVector;
   fVector.push_back(reco::Track());
   fVector.push_back(reco::Track());
   fVector.push_back(reco::Track());
   
   TClass* cls=TClass::GetClass("reco::TrackCollection");
   assert(0!=cls);
   
   FWEventItem item(&cm, 0,0,"Tracks", cls);
   cm.newItemSlot(&item);
   
   
   Listener listener;
   ItemListener iListener;
   //NOTE: have to pass a pointer to the listener else the bind will
   // create a copy of the listener and the original one will never
   // 'hear' any signal
   item.changed_.connect(boost::bind(&Listener::listen,&listener,_1));
   item.itemChanged_.connect(boost::bind(&ItemListener::listen,&iListener,_1));
   
   BOOST_CHECK(listener.nHeard_ ==0 );
   BOOST_CHECK(iListener.nHeard_ ==0 );
   cm.changed(FWModelId(&item,0));
   BOOST_CHECK(listener.nHeard_ ==1 );
   BOOST_CHECK(iListener.nHeard_ ==0 );

   listener.nHeard_ =0;
   cm.beginChanges();
   cm.changed(FWModelId(&item,0));
   BOOST_CHECK(listener.nHeard_ ==0 );
   cm.endChanges();
   BOOST_CHECK(listener.nHeard_ ==1 );
   
   //sending same ID twice should give only 1 message
   listener.nHeard_ =0;
   cm.beginChanges();
   cm.changed(FWModelId(&item,0));
   BOOST_CHECK(listener.nHeard_ ==0 );
   cm.changed(FWModelId(&item,0));
   BOOST_CHECK(listener.nHeard_ ==0 );
   cm.endChanges();
   BOOST_CHECK(listener.nHeard_ ==1 );

   listener.nHeard_ =0;
   cm.beginChanges();
   cm.changed(FWModelId(&item,1));
   BOOST_CHECK(listener.nHeard_ ==0 );
   cm.changed(FWModelId(&item,2));
   BOOST_CHECK(listener.nHeard_ ==0 );
   cm.endChanges();
   BOOST_CHECK(listener.nHeard_ ==2 );

   listener.nHeard_ =0;
   {
      FWChangeSentry sentry(cm);
      cm.changed(FWModelId(&item,1));
      BOOST_CHECK(listener.nHeard_ ==0 );
   }
   BOOST_CHECK(listener.nHeard_ ==1 );
   
   BOOST_CHECK(iListener.nHeard_ ==0 );
   item.setEvent(0);
   BOOST_CHECK(iListener.nHeard_ ==1 );

   iListener.nHeard_=0;
   {
      FWChangeSentry sentry(cm);
      item.setEvent(0);
      BOOST_CHECK(iListener.nHeard_ ==0 );
   }
   BOOST_CHECK(iListener.nHeard_ ==1 );
}
