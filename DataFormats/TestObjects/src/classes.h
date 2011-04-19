#include "DataFormats/Common/interface/SortedCollection.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/Common/interface/Wrapper.h"

#include "DataFormats/TestObjects/interface/OtherThingCollection.h"
#include "DataFormats/TestObjects/interface/ThingCollection.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include "DataFormats/TestObjects/interface/Thing.h"
#include "DataFormats/TestObjects/interface/ThingWithMerge.h"
#include "DataFormats/TestObjects/interface/ThingWithIsEqual.h"

#include "DataFormats/TestObjects/interface/StreamTestSimple.h"
#include "DataFormats/TestObjects/interface/StreamTestThing.h"
#include "DataFormats/TestObjects/interface/StreamTestTmpl.h"

#include "DataFormats/Common/interface/Holder.h"
#include "DataFormats/Common/interface/RefToBaseProd.h"

namespace {
struct dictionary {
  edm::Wrapper<edmtest::DummyProduct> dummyw12;
  edm::Wrapper<edmtest::IntProduct> dummyw13;
  edm::Wrapper<edmtest::TransientIntProduct> dummyw13t;
  edm::Wrapper<edmtest::DoubleProduct> dummyw14;
  edm::Wrapper<edmtest::StringProduct> dummyw15;
  edm::Wrapper<edmtest::SCSimpleProduct> dummyw16;
  edm::Wrapper<edmtest::OVSimpleProduct> dummyw17;
  edm::Wrapper<edmtest::OVSimpleDerivedProduct> dummyw17Derived;
  edm::Wrapper<edmtest::AVSimpleProduct> dummyw18;
  edm::Wrapper<edmtest::DSVSimpleProduct> dummyw19;
  edm::Wrapper<edmtest::DSVWeirdProduct> dummyw20;
  edm::Wrapper<edmtest::DSTVSimpleProduct> dummyw21;
  edm::Wrapper<edmtest::DSTVSimpleDerivedProduct> dummyw22;
  edm::Wrapper<edmtest::Int16_tProduct> dummyw23;
  edm::Wrapper<edmtest::Prodigal> dummyw24;

  edm::Wrapper<edmtest::Thing> dummy105;
  edm::Wrapper<edmtest::ThingWithMerge> dummy104;
  edm::Wrapper<edmtest::ThingWithIsEqual> dummy103;

  edmtest::ThingCollection dummy1;
  edmtest::OtherThingCollection dummy2;
  edm::Wrapper<edmtest::ThingCollection> dummy3;
  edm::Wrapper<edmtest::OtherThingCollection> dummy4;

  edmtestprod::Ord<edmtestprod::Simple> dummy20;
  edmtestprod::StreamTestTmpl<edmtestprod::Ord<edmtestprod::Simple> > dummy21;
  edm::Wrapper<edmtestprod::StreamTestTmpl<edmtestprod::Ord<edmtestprod::Simple> > > dummy22;
  std::vector<edmtestprod::Simple> dummy23;
  std::vector<edmtest::Simple> dummy231;
  edm::Wrapper<std::vector<edmtest::Simple> > dummy231w;
  edm::RefProd<std::vector<edmtest::Simple> > dummy232;
  edm::SortedCollection<edmtestprod::Simple,edm::StrictWeakOrdering<edmtestprod::Simple> > dummy24;
  edm::Wrapper<edm::SortedCollection<edmtestprod::Simple,edm::StrictWeakOrdering<edmtestprod::Simple> > > dummy25;
  edm::Wrapper<edmtestprod::StreamTestThing> dummy26;
  edm::Wrapper<edmtestprod::X0123456789012345678901234567890123456789012345678901234567890123456789012345678901> dummy27;
  edm::DetSet<edmtest::Sortable> x1;
  edm::DetSet<edmtest::Unsortable> x2;
  std::vector<edmtest::Sortable> x3;
  std::vector<edmtest::Unsortable> x4;

  edm::reftobase::Holder<edmtest::Thing,edm::Ref<std::vector<edmtest::Thing> > > bhThing;
  edm::RefToBaseProd<edmtest::Thing> rtbpThing;
  
  edm::Ptr<edmtest::Thing> ptrThing;
  edm::PtrVector<edmtest::Thing> ptrVecThing;
};
}
