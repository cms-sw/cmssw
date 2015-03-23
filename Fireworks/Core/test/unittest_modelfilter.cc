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
#include <boost/test/test_tools.hpp>
#include "TClass.h"

// user include files
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#define private public
#include "Fireworks/Core/interface/FWEventItem.h"
#undef private

#include "Fireworks/Core/interface/FWModelChangeManager.h"
#include "Fireworks/Core/interface/FWSelectionManager.h"

#include "Fireworks/Core/interface/FWItemAccessorBase.h"

#include "Fireworks/Core/interface/FWModelFilter.h"

//
// constants, enums and typedefs
//
namespace  {
   class TestAccessor : public FWItemAccessorBase {
   public:
      TestAccessor(const reco::TrackCollection* iCollection):
      m_collection(iCollection) {}
      virtual const void* modelData(int iIndex) const {return &((*m_collection)[iIndex]);}
      virtual const void* data() const {return m_collection;}
      virtual unsigned int size() const {return m_collection->size();}
      virtual const TClass* modelType() const {return TClass::GetClass("reco::Track");}
      virtual const TClass* type() const {return TClass::GetClass("reco::TrackCollection");}
      
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

BOOST_AUTO_TEST_CASE( itemfilter )
{
   FWModelChangeManager cm;
   
   FWSelectionManager sm(&cm);


   reco::TrackCollection fVector;
   fVector.push_back(reco::Track());
   fVector.push_back(reco::Track(20,20,reco::Track::Point(), 
                                 reco::Track::Vector(20,0,0), -1, reco::Track::CovarianceMatrix()));
   fVector.push_back(reco::Track());
   
   TClass* cls=TClass::GetClass("std::vector<reco::Track>");
   assert(0!=cls);
   
   fireworks::Context context(&cm,&sm,0,0,0);
   
   boost::shared_ptr<FWItemAccessorBase> accessor( new TestAccessor(&fVector));
   FWPhysicsObjectDesc pObj("Tracks",cls,"Tracks");
   FWEventItem item(&context, 0,accessor,pObj);

   cm.newItemSlot(&item);
   
   FWModelFilter filter("$.pt() > 10","reco::Track");
   BOOST_CHECK(not filter.passesFilter(item.modelData(0)));
   BOOST_CHECK(filter.passesFilter(item.modelData(1)));
   
   filter.setExpression(std::string());
   BOOST_CHECK(filter.passesFilter(item.modelData(0)));
   BOOST_CHECK(filter.passesFilter(item.modelData(1)));
}
