
#include <vector>

#include "FWCore/Framework/interface/TriggerResults.h"
#include "FWCore/Framework/interface/EventAux.h"
#include "FWCore/Framework/interface/EventProvenance.h"
#include "FWCore/Framework/interface/BranchDescription.h"
#include "FWCore/Framework/interface/BranchEntryDescription.h"
#include "FWCore/Framework/interface/ProductRegistry.h"
#include "FWCore/Framework/interface/BranchKey.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "FWCore/Framework/src/ToyProducts.h"
#include <map>

namespace{
namespace{
	edm::Wrapper<edmtest::DummyProduct> dummy;
	edm::Wrapper<edmtest::IntProduct> dummy1;
	edm::Wrapper<edmtest::DoubleProduct> dummy2;
	edm::Wrapper<edmtest::StringProduct> dummy3;
	edm::Wrapper<edmtest::SCSimpleProduct> dummy4;
	edm::Wrapper<edm::TriggerResults> dummy5;
}
}
