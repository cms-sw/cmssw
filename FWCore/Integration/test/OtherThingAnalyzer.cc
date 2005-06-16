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
      if (tcv.a == i && xv == i) {
        std::cout << " VECTOR ITEM " << i << " dereferenced successfully. " << std::endl;
      } else {
        std::cout << "ERROR: VECTOR ITEM " << i << " has incorrect value " << tc.a << '.' << std::endl;
      }
    }
  }
DEFINE_FWK_MODULE(OtherThingAnalyzer)
}
