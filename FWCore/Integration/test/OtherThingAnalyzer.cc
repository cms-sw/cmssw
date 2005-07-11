#include <iostream>
#include "FWCore/FWCoreIntegration/test/OtherThingAnalyzer.h"
#include "FWCore/FWCoreIntegration/test/OtherThing.h"
#include "FWCore/FWCoreIntegration/test/OtherThingCollection.h"
#include "FWCore/FWCoreIntegration/test/ThingCollection.h"
#include "FWCore/CoreFramework/interface/Event.h"
#include "FWCore/CoreFramework/interface/Handle.h"
#include "FWCore/CoreFramework/interface/MakerMacros.h"

namespace edmreftest {
  void OtherThingAnalyzer::analyze(edm::Event const& e, edm::EventSetup const&) {
    edm::Handle<OtherThingCollection> otherThings;
    e.getByLabel("OtherThing", "testUserTag", otherThings);
    std::cout << " --------------- next event ------------ " << std::endl;
    int i = 0;
    for (OtherThingCollection::const_iterator it = (*otherThings).begin(); it != (*otherThings).end(); ++it, ++i) {
      OtherThing const& otc = *it;
      Thing const & tc = *otc.ref;
      int const & x = otc.ref->a;
      if (tc.a == i && x == i) {
        std::cout << " ITEM " << i << " dereferenced successfully. " << std::endl;
      } else {
        std::cout << "ERROR: ITEM " << i << " has incorrect value " << tc.a << '.' << std::endl;
      }
      bool shouldBeTrue = otc.refVec[0] != otc.refVec[1];
      if (!shouldBeTrue) {
        std::cout << "ERROR: inequality has incorrect value" << std::endl;
      }
      shouldBeTrue = otc.refVec[0] == otc.refVec[0];
      if (!shouldBeTrue) {
        std::cout << "ERROR: equality has incorrect value" << std::endl;
      }
      shouldBeTrue = otc.refVec[0].isNonnull();
      if (!shouldBeTrue) {
        std::cout << "ERROR: non-null check has incorrect value" << std::endl;
      }
      shouldBeTrue = !(!otc.refVec[0]);
      if (!shouldBeTrue) {
        std::cout << "ERROR: ! has incorrect value" << std::endl;
      }
      shouldBeTrue = bool(otc.refVec[0]);
      if (!shouldBeTrue) {
        std::cout << "ERROR: bool() has incorrect value" << std::endl;
      }
      shouldBeTrue = !otc.refVec.empty();
      if (!shouldBeTrue) {
        std::cout << "ERROR: empty() has incorrect value" << std::endl;
      }
      shouldBeTrue = (otc.refVec == otc.refVec);
      if (!shouldBeTrue) {
        std::cout << "ERROR: RefVector equality has incorrect value" << std::endl;
      }
      shouldBeTrue = !(otc.refVec != otc.refVec);
      if (!shouldBeTrue) {
        std::cout << "ERROR: RefVector inequality has incorrect value" << std::endl;
      }
      Thing const & tcv = *otc.refVec[0];
      int const & xv = otc.refVec[0]->a;
      if (xv != tcv.a || xv != i) {
        std::cout << "ERROR: VECTOR ITEM 0 " << i << " has incorrect value " << tcv.a << '.' << std::endl;
      }
      Thing const & tcv1 = *otc.refVec[1];
      int const & xv1 = otc.refVec[1]->a;
      if (xv1 != tcv1.a || xv1 != 19-i) {
        std::cout << "ERROR: VECTOR ITEM 1 " << i << " has incorrect value " << tcv1.a << '.' << std::endl;
      }
      for (edm::RefVector<ThingCollection>::iterator it = otc.refVec.begin();
          it != otc.refVec.end(); ++it) {
        edm::Ref<ThingCollection> tcol = *it;
        Thing const & ti = **it;
        int const & xi = (*it)->a;
	if (xi != ti.a) {
          std::cout << "ERROR: iterator item " << ti.a << " " << xi << std::endl;
        } else if (it == otc.refVec.begin() && xi != i) {
          std::cout << "ERROR: iterator item 0" << xi << std::endl;
        } else if (it != otc.refVec.begin() && xi != 19-i) {
          std::cout << "ERROR: iterator item 1" << xi << std::endl;
        }
      }
      edm::RefVector<ThingCollection>::iterator it0 = otc.refVec.begin();
      int zero = (*it0)->a;
      edm::RefVector<ThingCollection>::iterator it1 = it0 + 1;
      int one = (*it1)->a;
      it1 = 1 + it0;
      int x1 = (*it1)->a;
      if (x1 != one) std::cout << "operator+ ITERATOR ERROR: " << x1 << " " << one << std::endl;
      it0 = it1 - 1;
      int x0 = (*it0)->a;
      if (x0 != zero) std::cout << "operator- ITERATOR ERROR: " << x0 << " " << zero << std::endl;
      x0 = (*(it0++))->a;
      if (x0 != zero) std::cout << "operator++ ITERATOR ERROR: " << x0 << " " << zero << std::endl;
      x1 = (*it0)->a;
      if (x1 != one) std::cout << "operator++ ITERATOR ERROR 2: " << x1 << " " << one << std::endl;
      x1 = (*(it0--))->a;
      if (x1 != one) std::cout << "operator-- ITERATOR ERROR: " << x1 << " " << one << std::endl;
      x0 = (*it0)->a;
      if (x0 != zero) std::cout << "operator-- ITERATOR ERROR 2: " << x0 << " " << zero << std::endl;
      x1 = it0[1]->a;
      if (x1 != one) std::cout << "operator[] ITERATOR ERROR: " << x1 << " " << one << std::endl;
      x1 = it1[0]->a;
      if (x1 != one) std::cout << "operator[] ITERATOR ERROR 2: " << x1 << " " << one << std::endl;
      std::cout << "EVENT " << it0->evtID() << std::endl;
    }
  }
}
using edmreftest::OtherThingAnalyzer;
DEFINE_FWK_MODULE(OtherThingAnalyzer)
