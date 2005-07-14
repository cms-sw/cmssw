#include "FWCore/Integration/test/ThingCollection.h"
#include "FWCore/Integration/test/OtherThingCollection.h"
#include "FWCore/EDProduct/interface/Wrapper.h"

namespace {
namespace {
	edmreftest::ThingCollection dummy1;
	edmreftest::OtherThingCollection dummy2;
	edm::Wrapper<edmreftest::ThingCollection> dummy3;
	edm::Wrapper<edmreftest::OtherThingCollection> dummy4;
}
}

