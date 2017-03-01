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
//

// system include files
#include <boost/test/auto_unit_test.hpp>
#include <boost/bind.hpp>
#include <boost/test/test_tools.hpp>
#include "TClass.h"

// user include files
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWModelChangeManager.h"
#include "Fireworks/Core/interface/FWSelectionManager.h"
#include "Fireworks/Core/interface/FWColorManager.h"
#include "Fireworks/Core/interface/FWEventItemsManager.h"
#include "Fireworks/Core/interface/FWConfiguration.h"

//
// constants, enums and typedefs
//
namespace {
   struct Listener {
      Listener(): nMessages_(0), item_(0) {}
      int nMessages_;
      const FWEventItem* item_;
      
      void reset() {
         nMessages_=0;
         item_=0;
      }
      void newItem(const FWEventItem* iItem) {
         ++nMessages_;
         item_=iItem;
      }
   };
}

BOOST_AUTO_TEST_CASE( eventitemmanager )
{
   FWModelChangeManager cm;
  
   FWSelectionManager sm(&cm);
   FWEventItemsManager eim(&cm);
   FWColorManager colm(&cm);
   colm.initialize();

   // !!!! Passing 0 for FWJobMetadataManager
   fireworks::Context context(&cm,&sm,&eim,&colm,0);
   eim.setContext(&context);
   
   Listener listener;
   //NOTE: have to pass a pointer to the listener else the bind will
   // create a copy of the listener and the original one will never
   // 'hear' any signal
   eim.newItem_.connect(boost::bind(&Listener::newItem,&listener,_1));

   TClass* cls=TClass::GetClass("std::vector<reco::Track>");
   assert(0!=cls);

   Color_t color1 = FWColorManager::getDefaultStartColorIndex() + 1;
   FWPhysicsObjectDesc tracks("Tracks",
                              cls,
                              "Tracks",
                              FWDisplayProperties(color1, true, 100),
                              "label",
                              "instance",
                              "proc");

   BOOST_REQUIRE(listener.nMessages_==0);
   BOOST_REQUIRE(eim.begin()==eim.end());

   eim.add(tracks);
   BOOST_CHECK(listener.nMessages_==1);
   BOOST_CHECK(eim.end()-eim.begin() == 1);
   const FWEventItem* item= *(eim.begin());
   BOOST_REQUIRE(item!=0);
   BOOST_CHECK(item == listener.item_);
   BOOST_CHECK(item->name() == "Tracks");
   BOOST_CHECK(item->type() == cls);
   BOOST_CHECK(item->defaultDisplayProperties().color() == color1);
   BOOST_CHECK(item->defaultDisplayProperties().isVisible());
   BOOST_CHECK(item->moduleLabel() == "label");
   BOOST_CHECK(item->productInstanceLabel() == "instance");
   BOOST_CHECK(item->processName() == "proc");

   FWConfiguration config;
   eim.addTo(config);
   
   eim.clearItems();   
   listener.reset();
   

   BOOST_REQUIRE(listener.nMessages_==0);
   BOOST_REQUIRE(eim.begin()==eim.end());

   eim.setFrom(config);
   
   BOOST_CHECK(listener.nMessages_==1);
   BOOST_CHECK(eim.end()-eim.begin() == 1);
   item= *(eim.begin());
   BOOST_REQUIRE(item!=0);
   BOOST_CHECK(item == listener.item_);
   BOOST_CHECK(item->name() == "Tracks");
   BOOST_CHECK(item->type() == cls);
   BOOST_CHECK(item->defaultDisplayProperties().color() == color1);
   BOOST_CHECK(item->defaultDisplayProperties().isVisible());
   BOOST_CHECK(item->moduleLabel() == "label");
   BOOST_CHECK(item->productInstanceLabel() == "instance");
   BOOST_CHECK(item->processName() == "proc");
}
