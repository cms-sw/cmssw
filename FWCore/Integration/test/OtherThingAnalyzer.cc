#include <iostream>
#include "FWCore/Integration/test/OtherThingAnalyzer.h"
#include "DataFormats/TestObjects/interface/OtherThing.h"
#include "DataFormats/TestObjects/interface/OtherThingCollection.h"
#include "DataFormats/TestObjects/interface/ThingCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace edmtest {
  void OtherThingAnalyzer::analyze(edm::Event const& e, edm::EventSetup const&) {
    doit(e.me(), std::string("testUserTag"));
  }

  void OtherThingAnalyzer::beginLuminosityBlock(edm::LuminosityBlock const& lb, edm::EventSetup const&) {
    doit(lb.me(), std::string("beginLumi"));
  }

  void OtherThingAnalyzer::endLuminosityBlock(edm::LuminosityBlock const& lb, edm::EventSetup const&) {
    doit(lb.me(), std::string("endLumi"));
  }

  void OtherThingAnalyzer::beginRun(edm::Run const& r, edm::EventSetup const&) {
    doit(r.me(), std::string("beginRun"));
  }

  void OtherThingAnalyzer::endRun(edm::Run const& r, edm::EventSetup const&) {
    doit(r.me(), std::string("endRun"));
  }

  void OtherThingAnalyzer::doit(edm::DataViewImpl const& dv, std::string const& label) {
    edm::Handle<OtherThingCollection> otherThings;
    dv.getByLabel("OtherThing", label, otherThings);
    edm::LogInfo("OtherThingAnalyzer") << " --------------- next event ------------ \n";
    int i = 0;
    for (OtherThingCollection::const_iterator it = (*otherThings).begin(), itEnd = (*otherThings).end();
        it != itEnd; ++it, ++i) {
      OtherThing const& otc = *it;
      ThingCollection const& tcoll = *otc.refProd;
      ThingCollection::size_type size1 = tcoll.size();
      ThingCollection::size_type size2 = otc.refProd->size();
      if (size1 == 0 || size1 != size2) {
        throw cms::Exception("Inconsistent Data", "OtherThingAnalyzer::analyze") << " RefProd size mismatch " << std::endl;
      }
      Thing const& tc = *otc.ref;
      int const& x = otc.ref->a;
      if (tc.a == i && x == i) {
        edm::LogInfo("OtherThingAnalyzer") << " ITEM " << i << " LABEL " << label << " dereferenced successfully.\n";
      } else {
        throw cms::Exception("Inconsistent Data", "OtherThingAnalyzer::analyze") << "ITEM " << i << " has incorrect value " << tc.a << '\n';
      }
      bool shouldBeTrue = otc.refVec[0] != otc.refVec[1];
      if (!shouldBeTrue) {
        throw cms::Exception("Inconsistent Data", "OtherThingAnalyzer::analyze") << "Inequality has incorrect value\n";
      }
      shouldBeTrue = otc.refVec[0] == otc.refVec[0];
      if (!shouldBeTrue) {
        throw cms::Exception("Inconsistent Data", "OtherThingAnalyzer::analyze") << "Equality has incorrect value\n";
      }
      shouldBeTrue = otc.refProd == otc.refProd;
      if (!shouldBeTrue) {
        throw cms::Exception("Inconsistent Data", "OtherThingAnalyzer::analyze") << "RefProd Equality has incorrect value\n";
      }
      shouldBeTrue = otc.refVec[0].isNonnull();
      if (!shouldBeTrue) {
        throw cms::Exception("Inconsistent Data", "OtherThingAnalyzer::analyze") << "Non-null check has incorrect value\n";
      }
      shouldBeTrue = otc.refProd.isNonnull();
      if (!shouldBeTrue) {
        throw cms::Exception("Inconsistent Data", "OtherThingAnalyzer::analyze") << "RefProd non-null check has incorrect value\n";
      }
      shouldBeTrue = !(!otc.refVec[0]);
      if (!shouldBeTrue) {
        throw cms::Exception("Inconsistent Data", "OtherThingAnalyzer::analyze") << "'!' has incorrect value\n";
      }
      shouldBeTrue = !(!otc.refProd);
      if (!shouldBeTrue) {
        throw cms::Exception("Inconsistent Data", "OtherThingAnalyzer::analyze") << "RefProd '!' has incorrect value\n";
      }
      shouldBeTrue = !otc.refVec.empty();
      if (!shouldBeTrue) {
        throw cms::Exception("Inconsistent Data", "OtherThingAnalyzer::analyze") << "empty() has incorrect value\n";
      }
      shouldBeTrue = (otc.refVec == otc.refVec);
      if (!shouldBeTrue) {
        throw cms::Exception("Inconsistent Data", "OtherThingAnalyzer::analyze") << "RefVector equality has incorrect value\n";
      }
      shouldBeTrue = !(otc.refVec != otc.refVec);
      if (!shouldBeTrue) {
        throw cms::Exception("Inconsistent Data", "OtherThingAnalyzer::analyze") << "RefVector inequality has incorrect value\n";
      }
      Thing const& tcv = *otc.refVec[0];
      int const& xv = otc.refVec[0]->a;
      if (xv != tcv.a || xv != i) {
        throw cms::Exception("Inconsistent Data", "OtherThingAnalyzer::analyze") << "VECTOR ITEM 0 " << i << " has incorrect value " << tcv.a << '\n';
      }
      Thing const& tcv1 = *otc.refVec[1];
      int const& xv1 = otc.refVec[1]->a;
      if (xv1 != tcv1.a || xv1 != 19-i) {
        throw cms::Exception("Inconsistent Data", "OtherThingAnalyzer::analyze") << "VECTOR ITEM 1 " << i << " has incorrect value " << tcv1.a << '\n';
      }
      for (edm::RefVector<ThingCollection>::iterator it = otc.refVec.begin(), itEnd = otc.refVec.end();
          it != itEnd; ++it) {
        edm::Ref<ThingCollection> tcol = *it;
        Thing const& ti = **it;
        int const& xi = (*it)->a;
	if (xi != ti.a) {
        throw cms::Exception("Inconsistent Data", "OtherThingAnalyzer::analyze") << "iterator item " << ti.a << " " << xi << '\n';
        } else if (it == otc.refVec.begin() && xi != i) {
        throw cms::Exception("Inconsistent Data", "OtherThingAnalyzer::analyze") << "iterator item 0" << xi << '\n';
        } else if (it != otc.refVec.begin() && xi != 19-i) {
        throw cms::Exception("Inconsistent Data", "OtherThingAnalyzer::analyze") << "iterator item 1" << xi << '\n';
        }
      }
      edm::RefVector<ThingCollection>::iterator it0 = otc.refVec.begin();
      int zero = (*it0)->a;
      edm::RefVector<ThingCollection>::iterator it1 = it0 + 1;
      int one = (*it1)->a;
      it1 = 1 + it0;
      int x1 = (*it1)->a;
      if (x1 != one) throw cms::Exception("Inconsistent Data", "OtherThingAnalyzer::analyze")
        << "operator+ iterator error: " << x1 << " " << one << '\n';
      it0 = it1 - 1;
      int x0 = (*it0)->a;
      if (x0 != zero) throw cms::Exception("Inconsistent Data", "OtherThingAnalyzer::analyze")
        << "operator- iterator error: " << x0 << " " << zero << '\n';
      x0 = (*(it0++))->a;
      if (x0 != zero) throw cms::Exception("Inconsistent Data", "OtherThingAnalyzer::analyze")
        << "operator++ iterator error: " << x0 << " " << zero << '\n';
      x1 = (*it0)->a;
      if (x1 != one) throw cms::Exception("Inconsistent Data", "OtherThingAnalyzer::analyze")
        << "operator++ iterator error 2: " << x1 << " " << one << '\n';
      x1 = (*(it0--))->a;
      if (x1 != one) throw cms::Exception("Inconsistent Data", "OtherThingAnalyzer::analyze")
        << "operator-- iterator error: " << x1 << " " << one << '\n';
      x0 = (*it0)->a;
      if (x0 != zero) throw cms::Exception("Inconsistent Data", "OtherThingAnalyzer::analyze")
        << "operator-- iterator error 2: " << x0 << " " << zero << '\n';
      x1 = it0[1]->a;
      if (x1 != one) throw cms::Exception("Inconsistent Data", "OtherThingAnalyzer::analyze")
        << "operator[] iterator error: " << x1 << " " << one << '\n';
      x1 = it1[0]->a;
      if (x1 != one) throw cms::Exception("Inconsistent Data", "OtherThingAnalyzer::analyze")
        << "operator[] iterator error 2: " << x1 << " " << one << '\n';
    }
  }
}
using edmtest::OtherThingAnalyzer;
DEFINE_FWK_MODULE(OtherThingAnalyzer);
