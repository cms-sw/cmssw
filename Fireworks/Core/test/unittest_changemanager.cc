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
// $Id: unittest_changemanager.cc,v 1.6 2012/08/03 18:20:28 wmtan Exp $
//

// system include files
#include <boost/test/auto_unit_test.hpp>
#include <boost/bind.hpp>
#include <boost/test/test_tools.hpp>

#include "TClass.h"
#include "Cintex/Cintex.h"

// user include files
#include "Fireworks/Core/interface/FWModelChangeManager.h"
#define private public
#include "Fireworks/Core/interface/FWEventItem.h"
#undef private
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "Fireworks/Core/interface/FWItemAccessorBase.h"

#include "FWCore/Utilities/interface/ObjectWithDict.h"


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
   
   class TestAccessor : public FWItemAccessorBase {
   public:
      TestAccessor(const reco::TrackCollection* iCollection):
      m_collection(iCollection) {}
      virtual const void* modelData(int iIndex) const {return &((*m_collection)[iIndex]);}
      virtual const void* data() const {return m_collection;}
      virtual unsigned int size() const {return m_collection->size();}
      virtual const TClass* modelType() const {return TClass::GetClass("reco::Track");}
      virtual const TClass* type() const {return TClass::GetClass("std::vector<reco::Track>");}
      
      virtual bool isCollection() const {return true;}
      
      ///override if id of an object should be different than the index
      //virtual std::string idForIndex(int iIndex) const;
      // ---------- member functions ---------------------------
      virtual void setData(const edm::ObjectWithDict& ) {}
      virtual void reset(){}
      
   private:
      const reco::TrackCollection* m_collection;
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
   
   TClass* cls=TClass::GetClass("std::vector<reco::Track>");
   assert(0!=cls);
   
   fireworks::Context context(&cm,0,0,0,0);
   
   boost::shared_ptr<FWItemAccessorBase> accessor( new TestAccessor(&fVector));
   FWPhysicsObjectDesc pObj("Tracks",cls,"Tracks");
   
   FWEventItem item(&context, 0,accessor,pObj);
   //hack to force update of data
   edm::ObjectWithDict dummy;
   item.setData(dummy);
   
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
