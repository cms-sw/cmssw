#include <vector>
#include "FWCore/CoreFramework/interface/EventAux.h"
#include "FWCore/CoreFramework/interface/Provenance.h"
#include "FWCore/CoreFramework/interface/EventProvenance.h"
#include "FWCore/EDProduct/interface/Wrapper.h"
#include "FWCore/CoreFramework/src/ToyProducts.h"

namespace{
namespace{
	edm::Wrapper<edmtest::DummyProduct> dummy;
	edm::Wrapper<edmtest::IntProduct> dummy1;
	edm::Wrapper<edmtest::DoubleProduct> dummy2;
	edm::Wrapper<edmtest::StringProduct> dummy3;
}
}
