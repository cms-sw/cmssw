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
    e.getByLabel("OtherThing", otherThings);
    std::cout << " --------------- next event ------------ " << std::endl;
    for (int i = 0; i < 20; ++i) {
      OtherThing const& otc =(*otherThings)[i];
      Thing const & tc = *otc.ref;
      int const & x = otc.ref->a;
      if (tc.a == i && x == i) {
        std::cout << " ITEM " << i << " dereferenced successfully. " << std::endl;
      } else {
        std::cout << "ERROR: ITEM " << i << " has incorrect value " << tc.a << '.' << std::endl;
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
      for (edm::RefVector<ThingCollection>::const_iterator it = otc.refVec.begin();
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
    }
  }
DEFINE_FWK_MODULE(OtherThingAnalyzer)
}
