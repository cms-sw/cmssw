#include <vector>
#include "FWCore/Framework/interface/EventAux.h"
#include "FWCore/Framework/interface/Provenance.h"
#include "FWCore/Framework/interface/EventProvenance.h"
#include "FWCore/EDProduct/interface/Wrapper.h"
#include "FWCore/Framework/src/ToyProducts.h"

namespace{
namespace{
	edm::Wrapper<edmtest::DummyProduct> dummy;
	edm::Wrapper<edmtest::IntProduct> dummy1;
	edm::Wrapper<edmtest::DoubleProduct> dummy2;
	edm::Wrapper<edmtest::StringProduct> dummy3;
}
}
