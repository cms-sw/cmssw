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
// $Id: unittest_modelexpressionselector.cc,v 1.1 2008/01/24 00:30:30 chrjones Exp $
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
#include "Fireworks/Core/interface/FWModelExpressionSelector.h"

//
// constants, enums and typedefs
//
BOOST_AUTO_TEST_CASE( modelexpressionselector )
{
   ROOT::Cintex::Cintex::Enable();
   FWModelChangeManager cm;
   
   FWSelectionManager sm(&cm);


   reco::TrackCollection fVector;
   fVector.push_back(reco::Track());
   fVector.push_back(reco::Track(20,20,reco::Track::Point(), 
                                 reco::Track::Vector(20,0,0), -1, reco::Track::CovarianceMatrix()));
   fVector.push_back(reco::Track());
   
   TClass* cls=TClass::GetClass("reco::TrackCollection");
   assert(0!=cls);
   
   FWEventItem item(&cm, &sm,0,"Tracks", cls);
   //cheat
   item.setData(&fVector);

   cm.newItemSlot(&item);
   
   FWModelExpressionSelector selector;
   {      
      FWChangeSentry sentry(cm);
      selector.select(&item, "$.pt() > 10");
   }
   BOOST_CHECK(1==sm.selected().size());
   BOOST_CHECK(not item.modelInfo(0).m_isSelected);
   BOOST_CHECK(item.modelInfo(1).m_isSelected);
   BOOST_CHECK(not item.modelInfo(2).m_isSelected);
   
}
