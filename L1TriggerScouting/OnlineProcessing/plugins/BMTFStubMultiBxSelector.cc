#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/StreamID.h"

// L1 scouting
#include "DataFormats/L1Scouting/interface/L1ScoutingBMTFStub.h"
#include "DataFormats/L1Scouting/interface/OrbitCollection.h"
#include "L1TriggerScouting/Utilities/interface/conversion.h"

#include <iostream>
#include <memory>
#include <utility>
#include <vector>
#include <string>
#include <numeric>
#include <set>

using namespace l1ScoutingRun3;

namespace l1ScoutingRun3 {
  enum BMTFSelectorConditionType { Simple, Wheel };
}

class BMTFStubMultiBxSelector : public edm::stream::EDProducer<> {
public:
  explicit BMTFStubMultiBxSelector(const edm::ParameterSet&);
  ~BMTFStubMultiBxSelector() override {}
  static void fillDescriptions(edm::ConfigurationDescriptions&);

  unsigned makeWheelPattern(const edm::Handle<OrbitCollection<BMTFStub>>& stubs, unsigned bx);
  bool windowHasCloseWheels(const std::vector<unsigned>& bxs);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  // tokens for scouting data
  edm::EDGetTokenT<OrbitCollection<l1ScoutingRun3::BMTFStub>> stubsTokenData_;

  // Condition
  BMTFSelectorConditionType condition_;

  // Selection thresholds
  unsigned bxWindowLength_;
  unsigned minNBMTFStub_;
};

BMTFStubMultiBxSelector::BMTFStubMultiBxSelector(const edm::ParameterSet& iPSet)
    : stubsTokenData_(consumes(iPSet.getParameter<edm::InputTag>("stubsTag"))),
      bxWindowLength_(iPSet.getParameter<unsigned>("bxWindowLength")),
      minNBMTFStub_(iPSet.getParameter<unsigned>("minNBMTFStub")) {
  std::string conditionStr = iPSet.getParameter<std::string>("condition");
  if (conditionStr == "simple")
    condition_ = Simple;
  else if (conditionStr == "wheel")
    condition_ = Wheel;
  else
    throw cms::Exception("BMTFStubMultiBxSelector::BMTFStubMultiBxSelector")
        << "Condition '" << conditionStr << "' not supported or not found";

  produces<std::vector<unsigned>>("SelBx").setBranchAlias("MultiBxStubsSelectedBx");
}

// ------------ method called for each ORBIT  ------------
void BMTFStubMultiBxSelector::produce(edm::Event& iEvent, const edm::EventSetup&) {
  edm::Handle<OrbitCollection<BMTFStub>> stubsCollection;

  iEvent.getByToken(stubsTokenData_, stubsCollection);

  // This is necessary to have a clean MultiBxStubsSelectedBx collection
  // Indeed, if there is overlap betweem two valid windows, there will be duplicated BXs
  std::unique_ptr<std::set<unsigned>> uniqueStubSelectedBxs(new std::set<unsigned>);

  // Loop over valid bunch crossings
  std::vector<unsigned> vNumStubBx(bxWindowLength_, 0);
  std::vector<unsigned> vWheelPatternBx(bxWindowLength_, 0);
  for (const unsigned& bx : stubsCollection->getFilledBxs()) {
    // Get number of stubs in current window [BX-ws+1, BX]
    for (unsigned i = 0; i < std::min(bxWindowLength_, bx); ++i)
      vNumStubBx[i] = stubsCollection->getBxSize(bx - i);

    // Get number of stubs in current BX (last of the window) and enque
    unsigned numStubsWindow = std::reduce(vNumStubBx.begin(), vNumStubBx.end());

    // Simple condition: just number of stubs (already checked)
    if (condition_ == Simple) {
      // If there are enough stubs in window...
      if (numStubsWindow >= minNBMTFStub_) {
        // ...loop in window to add BXs (std::min to include edge case of first BXs)
        for (unsigned i = 0; i < std::min(bxWindowLength_, bx); ++i)
          uniqueStubSelectedBxs->insert(bx - i);
      }
    }
    // Wheel condition: enough longitudinally "neighbouring" stubs
    else if (condition_ == Wheel) {
      for (unsigned i = 0; i < std::min(bxWindowLength_, bx); ++i) {
        // Prepare pattern and add to window vector
        vWheelPatternBx[i] = makeWheelPattern(stubsCollection, bx - i);
      }

      // Check if there are stubs in different BXs with neighbouring wheels
      // For example (with window of 3 BXs, neighbouring condition abs(wheel_pair) <= 1)
      // BX-2         BX-1        BX0
      // s0(wh = 2)   s0(wh=0)    s1(wh=-2)
      // s1(wh = 1)
      //
      // => valid window! (neighbouring stubs s1 in BX-2 and s0 in BX-1, assuming nStub threshold is satisfied)
      bool validWindow = windowHasCloseWheels(vWheelPatternBx) && (numStubsWindow >= minNBMTFStub_);

      // If window is valid....
      if (validWindow) {
        // ...loop in window to add BXs (std::min to include edge case of first BXs)
        for (unsigned i = 0; i < std::min(bxWindowLength_, bx); ++i)
          uniqueStubSelectedBxs->insert(bx - i);
      }
    }
  }  // end orbit loop

  // Convert set of selected BXs to a vector and put collection in event content
  std::unique_ptr<std::vector<unsigned>> stubSelectedBx =
      std::make_unique<std::vector<unsigned>>(uniqueStubSelectedBxs->begin(), uniqueStubSelectedBxs->end());
  iEvent.put(std::move(stubSelectedBx), "SelBx");
}

unsigned BMTFStubMultiBxSelector::makeWheelPattern(const edm::Handle<OrbitCollection<BMTFStub>>& stubs, unsigned bx) {
  // 5 wheel numbers + 2 to handle boundaries in a more comfortable way:
  // 0b         0   X   X   X   X   X   0
  // wheels   bnd  +2  +1   0  -1  -2 bnd
  // position   6   5   4   3   2   1   0
  unsigned wheelPatternBx = 0;
  for (const auto& s : stubs->bxIterator(bx))
    wheelPatternBx |= (1 << (s.wheel() + 3));
  return wheelPatternBx;
}

bool BMTFStubMultiBxSelector::windowHasCloseWheels(const std::vector<unsigned>& bxs) {
  for (size_t i = 0; i < std::size(bxs); ++i) {
    for (size_t j = i + 1; j < std::size(bxs); ++j) {
      // Assume for example that we have the following patterns
      //          bwwwwwb (b = boundary, w = wheel fields)
      // BX-2 : 0b0010000 (stub with wheel=+2)
      // BX-1 : 0b0001000 (stub with wheel=+1)
      // Bitwise or comparison:
      // BX-2 : 0b0010000
      // BX-1 : 0b0011100
      //            bsb   (s = stub wheel, b = boundary wheels)
      unsigned compare = bxs[i] & ((bxs[j] << 1) | bxs[j] | (bxs[j] >> 1));
      bool checkWindow = compare & 0b0111110;
      if (checkWindow)
        return true;
    }
  }
  return false;
}

void BMTFStubMultiBxSelector::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(BMTFStubMultiBxSelector);
