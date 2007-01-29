#include "DataFormats/Common/interface/SortedCollection.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/Common/interface/Wrapper.h"

#include "DataFormats/TestObjects/interface/OtherThingCollection.h"
#include "DataFormats/TestObjects/interface/ThingCollection.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"

#include "DataFormats/TestObjects/interface/StreamTestSimple.h"
#include "DataFormats/TestObjects/interface/StreamTestThing.h"
#include "DataFormats/TestObjects/interface/StreamTestTmpl.h"

namespace { namespace {
  edm::Wrapper<edmtest::DummyProduct> dummyw12;
  edm::Wrapper<edmtest::IntProduct> dummyw13;
  edm::Wrapper<edmtest::DoubleProduct> dummyw14;
  edm::Wrapper<edmtest::StringProduct> dummyw15;
  edm::Wrapper<edmtest::SCSimpleProduct> dummyw16;
  edm::Wrapper<edmtest::OVSimpleProduct> dummyw17;
  edm::Wrapper<edmtest::AVSimpleProduct> dummyw18;
  edm::Wrapper<edmtest::DSVSimpleProduct> dummyw19;
  edm::Wrapper<edmtest::DSVWeirdProduct> dummyw20;

  edmtest::ThingCollection dummy1;
  edmtest::OtherThingCollection dummy2;
  edm::Wrapper<edmtest::ThingCollection> dummy3;
  edm::Wrapper<edmtest::OtherThingCollection> dummy4;

  edmtestprod::Ord<edmtestprod::Simple> dummy20;
  edmtestprod::StreamTestTmpl<edmtestprod::Ord<edmtestprod::Simple> > dummy21;
  edm::Wrapper<edmtestprod::StreamTestTmpl<edmtestprod::Ord<edmtestprod::Simple> > > dummy22;
  std::vector<edmtestprod::Simple> dummy23;
  edm::SortedCollection<edmtestprod::Simple,edm::StrictWeakOrdering<edmtestprod::Simple> > dummy24;
  edm::Wrapper<edm::SortedCollection<edmtestprod::Simple,edm::StrictWeakOrdering<edmtestprod::Simple> > > dummy25;
  edm::Wrapper<edmtestprod::StreamTestThing> dummy26;
  edm::Wrapper<edmtestprod::X0123456789012345678901234567890123456789012345678901234567890123456789012345678901> dummy27;
}}
