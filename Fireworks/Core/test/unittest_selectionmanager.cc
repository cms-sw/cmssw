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
//#include <boost/test/auto_unit_test.hpp>

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include "TClass.h"

// user include files
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/Reflection/interface/ObjectWithDict.h"
#include "Fireworks/Core/interface/FWEventItem.h"

#include "Fireworks/Core/interface/FWModelChangeManager.h"

#include "Fireworks/Core/interface/FWSelectionManager.h"

#include "Fireworks/Core/interface/FWItemAccessorBase.h"

//
// constants, enums and typedefs
//
namespace {
  struct Listener {
    Listener() : nHeard_(0), nMessages_(0) {}
    int nHeard_;
    int nMessages_;

    void reset() {
      nHeard_ = 0;
      nMessages_ = 0;
    }
    void listen(const FWSelectionManager& iSM) {
      nHeard_ += iSM.selected().size();
      ++nMessages_;
    }
  };

  class TestAccessor : public FWItemAccessorBase {
  public:
    TestAccessor(const reco::TrackCollection* iCollection) : m_collection(iCollection) {}
    virtual const void* modelData(int iIndex) const { return &((*m_collection)[iIndex]); }
    virtual const void* data() const { return m_collection; }
    virtual unsigned int size() const { return m_collection->size(); }
    virtual const TClass* modelType() const { return TClass::GetClass("reco::Track"); }
    virtual const TClass* type() const { return TClass::GetClass("reco::TrackCollection"); }

    virtual bool isCollection() const { return true; }

    ///override if id of an object should be different than the index
    //virtual std::string idForIndex(int iIndex) const;
    // ---------- member functions ---------------------------
    virtual void setData(const edm::ObjectWithDict&) {}
    virtual void reset() {}

  private:
    const reco::TrackCollection* m_collection;
  };

}  // namespace

BOOST_AUTO_TEST_CASE(selectionmanager) {
  FWModelChangeManager cm;

  FWSelectionManager sm(&cm);

  Listener listener;
  //NOTE: have to pass a pointer to the listener else the bind will
  // create a copy of the listener and the original one will never
  // 'hear' any signal
  sm.selectionChanged_.connect(std::bind(&Listener::listen, &listener, std::placeholders::_1));

  reco::TrackCollection fVector;
  fVector.push_back(reco::Track());
  fVector.push_back(reco::Track());
  fVector.push_back(reco::Track());

  TClass* cls = TClass::GetClass("std::vector<reco::Track>");
  assert(0 != cls);

  fireworks::Context context(&cm, &sm, 0, 0, 0);

  auto accessor = std::make_shared<TestAccessor>(&fVector);
  FWPhysicsObjectDesc pObj("Tracks", cls, "Tracks");

  FWEventItem item(&context, 0, accessor, pObj);
  //hack to force update of data
  edm::ObjectWithDict dummy;
  item.setData(dummy);

  cm.newItemSlot(&item);
  BOOST_CHECK(listener.nHeard_ == 0);
  BOOST_CHECK(listener.nMessages_ == 0);
  BOOST_CHECK(sm.selected().empty());

  item.select(0);
  BOOST_CHECK(listener.nHeard_ == 1);
  BOOST_CHECK(listener.nMessages_ == 1);
  BOOST_CHECK(sm.selected().size() == 1);
  BOOST_CHECK(item.modelInfo(0).m_isSelected);

  //selecting the same should not change the state
  item.select(0);
  BOOST_CHECK(listener.nHeard_ == 1);
  BOOST_CHECK(listener.nMessages_ == 1);
  BOOST_CHECK(sm.selected().size() == 1);

  listener.reset();
  item.unselect(0);
  BOOST_CHECK(listener.nHeard_ == 0);
  BOOST_CHECK(listener.nMessages_ == 1);
  BOOST_CHECK(sm.selected().empty());
  BOOST_CHECK(not item.modelInfo(0).m_isSelected);

  item.select(0);
  BOOST_CHECK(sm.selected().size() == 1);
  listener.reset();
  sm.clearSelection();
  BOOST_CHECK(listener.nHeard_ == 0);
  BOOST_CHECK(listener.nMessages_ == 1);
  BOOST_CHECK(sm.selected().empty());
  BOOST_CHECK(not item.modelInfo(0).m_isSelected);
  {
    //test waiting
    item.select(0);
    listener.reset();

    FWChangeSentry sentry(cm);
    sm.clearSelection();
    BOOST_CHECK(listener.nHeard_ == 0);
    BOOST_CHECK(listener.nMessages_ == 0);
    BOOST_CHECK(sm.selected().size() == 1);
    item.select(1);
    BOOST_CHECK(listener.nHeard_ == 0);
    BOOST_CHECK(listener.nMessages_ == 0);
    BOOST_CHECK(sm.selected().size() == 1);
    item.select(2);
    BOOST_CHECK(listener.nHeard_ == 0);
    BOOST_CHECK(listener.nMessages_ == 0);
    BOOST_CHECK(sm.selected().size() == 1);
  }
  BOOST_CHECK(listener.nHeard_ == 2);
  BOOST_CHECK(listener.nMessages_ == 1);
  BOOST_CHECK(sm.selected().size() == 2);
  BOOST_CHECK(not item.modelInfo(0).m_isSelected);
  BOOST_CHECK(item.modelInfo(1).m_isSelected);
  BOOST_CHECK(item.modelInfo(2).m_isSelected);

  sm.clearSelection();
  item.select(1);
  item.select(2);
  BOOST_CHECK(sm.selected().size() == 2);
  item.setEvent(0);
  BOOST_CHECK(sm.selected().size() == 0);
}
