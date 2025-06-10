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

class BMTFStubMultiBxSelector : public edm::stream::EDProducer<> {
public:
  explicit BMTFStubMultiBxSelector(const edm::ParameterSet&);
  ~BMTFStubMultiBxSelector() override {}
  static void fillDescriptions(edm::ConfigurationDescriptions&);

  bool windowHasCloseWheels(const std::vector<std::set<int>>& sets, int threshold = 1);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  // tokens for scouting data
  edm::EDGetTokenT<OrbitCollection<l1ScoutingRun3::BMTFStub>> stubsTokenData_;

  // Condition
  std::string condition_;

  // Selection thresholds
  unsigned bxWindowLength_;
  unsigned minNBMTFStub_;
};

BMTFStubMultiBxSelector::BMTFStubMultiBxSelector(const edm::ParameterSet& iPSet)
    : stubsTokenData_(consumes(iPSet.getParameter<edm::InputTag>("stubsTag"))),
      condition_(iPSet.getParameter<std::string>("condition")),
      bxWindowLength_(iPSet.getParameter<unsigned>("bxWindowLength")),
      minNBMTFStub_(iPSet.getParameter<unsigned>("minNBMTFStub"))
{
  std::vector<std::string> vConditions = {"simple", "wheel"};
  if (std::find(vConditions.begin(), vConditions.end(), condition_) == vConditions.end())
    throw cms::Exception("BMTFStubMultiBxSelector::BMTFStubMultiBxSelector")
      << "Condition '" << condition_ << "' not supported or not found";

  produces<std::vector<unsigned>>("SelBx").setBranchAlias("MultiBxStubsSelectedBx");
}

// ------------ method called for each ORBIT  ------------
void BMTFStubMultiBxSelector::produce(edm::Event& iEvent, const edm::EventSetup&) {
  edm::Handle<OrbitCollection<l1ScoutingRun3::BMTFStub>> stubsCollection;

  iEvent.getByToken(stubsTokenData_, stubsCollection);

  // This is necessary to have a clean MultiBxStubsSelectedBx collection
  // Indeed, if there is overlap betweem two valid windows, there will be duplicated BXs
  std::unique_ptr<std::set<unsigned>> uniqueStubSelectedBxs(new std::set<unsigned>);

  // Loop over valid bunch crossings
  for (const unsigned& bx : stubsCollection->getFilledBxs()) {
    if (bx < bxWindowLength_) continue;

    // Get number of stubs in every BX of the window and place in vector
    unsigned numStubsWindow = 0;
    std::vector<int> vNumStubBx(bxWindowLength_, 0);
    for (unsigned i = 0; i < bxWindowLength_; ++i)
      vNumStubBx[i] = stubsCollection->getBxSize(bx-i);

    // Sum elements of nStub vector
    numStubsWindow = std::reduce(vNumStubBx.begin(), vNumStubBx.end());

    // Not enough stubs in window, whatever the condition.
    if (numStubsWindow < minNBMTFStub_)
      continue;

    // Simple condition: just number of stubs (already checked)
    if (condition_=="simple") {
      for (unsigned i = 0; i < bxWindowLength_; ++i)
        uniqueStubSelectedBxs->insert(bx-i);
    }
    // Wheel condition: enough longitudinally "neighbouring" stubs
    else if (condition_=="wheel") {
      bool validWindow = false;

      // Find unique values for wheels in all BXs of window
      std::vector<std::set<int>> vWheelBx(bxWindowLength_);

      // Fill vector of sets of wheels (one element/set per BX in window)
      for (unsigned i = 0; i < bxWindowLength_; ++i)
        for (const auto& s : stubsCollection->bxIterator(bx-i)) vWheelBx[i].insert(s.wheel());

      // Check if there are stubs in different BXs with neighbouring wheels
      // For example (with window of 3 BXs, neighbouring condition abs(wheel_pair) <= 1)
      // BX-2         BX-1        BX0
      // s0(wh = 2)   s0(wh=0)    s1(wh=-2)
      // s1(wh = 1)
      //
      // => valid window! (neighbouring stubs s1 in BX-2 and s0 in BX-1)
      validWindow = windowHasCloseWheels(vWheelBx, 1);

      // If window is valid, add BXs of window to
      if (validWindow) {
        for (unsigned i = 0; i < bxWindowLength_; ++i)
          uniqueStubSelectedBxs->insert(bx-i);
      }
    }
  }  // end orbit loop

  // Convert set of selected BXs to a vector and put collection in event content
  std::unique_ptr<std::vector<unsigned>> stubSelectedBx = std::make_unique<std::vector<unsigned>>(uniqueStubSelectedBxs->begin(), uniqueStubSelectedBxs->end());
  iEvent.put(std::move(stubSelectedBx), "SelBx");
}

bool BMTFStubMultiBxSelector::windowHasCloseWheels(const std::vector<std::set<int>>& sets, int threshold) {
  for (size_t i = 0; i < std::size(sets); ++i) {
    for (size_t j = i + 1; j < std::size(sets); ++j) {
      for (int iWh : sets[i]) {
        for (int jWh : sets[j]) {
          if (std::abs(iWh - jWh) <= threshold) {
            return true;
          }
        }
      }
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
