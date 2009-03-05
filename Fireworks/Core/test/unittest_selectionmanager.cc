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
// $Id: unittest_selectionmanager.cc,v 1.4 2008/01/25 01:54:07 chrjones Exp $
//

// system include files
#include <boost/test/auto_unit_test.hpp>
#include <boost/bind.hpp>
#include <boost/test/test_tools.hpp>
#include "TClass.h"
#include "Cintex/Cintex.h"

// user include files
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#define private public
#include "Fireworks/Core/interface/FWEventItem.h"
#undef private

#include "Fireworks/Core/interface/FWModelChangeManager.h"

#include "Fireworks/Core/interface/FWSelectionManager.h"

//
// constants, enums and typedefs
//
namespace {
   struct Listener {
      Listener(): nHeard_(0),nMessages_(0) {}
      int nHeard_;
      int nMessages_;
      
      void reset() {
         nHeard_=0;
         nMessages_=0;
      }
      void listen(const FWSelectionManager& iSM) {
         nHeard_  += iSM.selected().size();
         ++nMessages_;
      }
   };
}

BOOST_AUTO_TEST_CASE( selectionmanager )
{
   ROOT::Cintex::Cintex::Enable();
   FWModelChangeManager cm;
   
   FWSelectionManager sm(&cm);

   Listener listener;
   //NOTE: have to pass a pointer to the listener else the bind will
   // create a copy of the listener and the original one will never
   // 'hear' any signal
   sm.selectionChanged_.connect(boost::bind(&Listener::listen,&listener,_1));

   reco::TrackCollection fVector;
   fVector.push_back(reco::Track());
   fVector.push_back(reco::Track());
   fVector.push_back(reco::Track());
   
   TClass* cls=TClass::GetClass("reco::TrackCollection");
   assert(0!=cls);
   
   FWEventItem item(&cm, &sm,0,"Tracks", cls);
   //cheat
   item.setData(&fVector);

   cm.newItemSlot(&item);
   BOOST_CHECK(listener.nHeard_ ==0 );
   BOOST_CHECK(listener.nMessages_==0);
   BOOST_CHECK(sm.selected().empty());
   
   item.select(0);
   BOOST_CHECK(listener.nHeard_==1);
   BOOST_CHECK(listener.nMessages_==1);
   BOOST_CHECK(sm.selected().size() == 1);
   BOOST_CHECK(item.modelInfo(0).m_isSelected);
   
   //selecting the same should not change the state
   item.select(0);
   BOOST_CHECK(listener.nHeard_==1);
   BOOST_CHECK(listener.nMessages_==1);
   BOOST_CHECK(sm.selected().size() == 1);

   listener.reset();
   item.unselect(0);
   BOOST_CHECK(listener.nHeard_ ==0 );
   BOOST_CHECK(listener.nMessages_==1);
   BOOST_CHECK(sm.selected().empty());
   BOOST_CHECK(not item.modelInfo(0).m_isSelected);

   item.select(0);
   BOOST_CHECK(sm.selected().size() == 1);
   listener.reset();
   sm.clearSelection();
   BOOST_CHECK(listener.nHeard_==0);
   BOOST_CHECK(listener.nMessages_==1);
   BOOST_CHECK(sm.selected().empty());
   BOOST_CHECK(not item.modelInfo(0).m_isSelected);
   {
      //test waiting
      item.select(0);
      listener.reset();
      
      FWChangeSentry sentry(cm);
      sm.clearSelection();
      BOOST_CHECK(listener.nHeard_ ==0 );
      BOOST_CHECK(listener.nMessages_==0);
      BOOST_CHECK(sm.selected().size()==1);
      item.select(1);
      BOOST_CHECK(listener.nHeard_ ==0 );
      BOOST_CHECK(listener.nMessages_==0);
      BOOST_CHECK(sm.selected().size()==1);
      item.select(2);
      BOOST_CHECK(listener.nHeard_ ==0 );
      BOOST_CHECK(listener.nMessages_==0);
      BOOST_CHECK(sm.selected().size()==1);      
   }
   BOOST_CHECK(listener.nHeard_ ==2 );
   BOOST_CHECK(listener.nMessages_==1);
   BOOST_CHECK(sm.selected().size()==2);      
   BOOST_CHECK(not item.modelInfo(0).m_isSelected);
   BOOST_CHECK(item.modelInfo(1).m_isSelected);
   BOOST_CHECK(item.modelInfo(2).m_isSelected);
   
   sm.clearSelection();
   item.select(1);
   item.select(2);
   BOOST_CHECK(sm.selected().size()==2);
   item.setEvent(0);
   BOOST_CHECK(sm.selected().size()==0);
}
