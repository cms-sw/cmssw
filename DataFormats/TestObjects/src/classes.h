#include "DataFormats/Common/interface/SortedCollection.h"
#include "DataFormats/Common/interface/Wrapper.h"

#include "DataFormats/TestObjects/interface/OtherThingCollection.h"
#include "DataFormats/TestObjects/interface/ThingCollection.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"

namespace { namespace {
  edm::Wrapper<edmtest::DummyProduct> dummyw12;
  edm::Wrapper<edmtest::IntProduct> dummyw13;
  edm::Wrapper<edmtest::DoubleProduct> dummyw14;
  edm::Wrapper<edmtest::StringProduct> dummyw15;
  edm::Wrapper<edmtest::SCSimpleProduct> dummyw16;

  edmtest::ThingCollection dummy1;
  edmtest::OtherThingCollection dummy2;
  edm::Wrapper<edmtest::ThingCollection> dummy3;
  edm::Wrapper<edmtest::OtherThingCollection> dummy4;
}}
