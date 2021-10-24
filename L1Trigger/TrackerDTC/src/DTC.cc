#include "L1Trigger/TrackerDTC/interface/DTC.h"

#include <vector>
#include <iterator>
#include <algorithm>
#include <numeric>

using namespace std;
using namespace edm;
using namespace tt;

namespace trackerDTC {

  DTC::DTC(const ParameterSet& iConfig,
           const Setup* setup,
           const LayerEncoding* layerEncoding,
           int dtcId,
           const std::vector<std::vector<TTStubRef>>& stubsDTC)
      : setup_(setup),
        enableTruncation_(iConfig.getParameter<bool>("EnableTruncation")),
        region_(dtcId / setup->numDTCsPerRegion()),
        board_(dtcId % setup->numDTCsPerRegion()),
        modules_(setup->dtcModules(dtcId)),
        input_(setup->dtcNumRoutingBlocks(), Stubss(setup->dtcNumModulesPerRoutingBlock())),
        lost_(setup->numOverlappingRegions()) {
    // count number of stubs on this dtc
    auto acc = [](int& sum, const vector<TTStubRef>& stubsModule) { return sum += stubsModule.size(); };
    const int nStubs = accumulate(stubsDTC.begin(), stubsDTC.end(), 0, acc);
    stubs_.reserve(nStubs);
    // convert and assign Stubs to DTC routing block channel
    for (int modId = 0; modId < setup->numModulesPerDTC(); modId++) {
      const vector<TTStubRef>& ttStubRefs = stubsDTC[modId];
      if (ttStubRefs.empty())
        continue;
      // Module which produced this ttStubRefs
      SensorModule* module = modules_.at(modId);
      // DTC routing block id [0-1]
      const int blockId = modId / setup->dtcNumModulesPerRoutingBlock();
      // DTC routing blockc  channel id [0-35]
      const int channelId = modId % setup->dtcNumModulesPerRoutingBlock();
      // convert TTStubs and fill input channel
      Stubs& stubs = input_[blockId][channelId];
      for (const TTStubRef& ttStubRef : ttStubRefs) {
        stubs_.emplace_back(iConfig, setup, layerEncoding, module, ttStubRef);
        Stub& stub = stubs_.back();
        if (stub.valid())
          // passed pt and eta cut
          stubs.push_back(&stub);
      }
      // sort stubs by bend
      sort(stubs.begin(), stubs.end(), [](Stub* lhs, Stub* rhs) { return abs(lhs->bend()) < abs(rhs->bend()); });
      // truncate stubs if desired
      if (!enableTruncation_ || (int)stubs.size() <= setup->numFramesFE())
        continue;
      // begin of truncated stubs
      const auto limit = next(stubs.begin(), setup->numFramesFE());
      // copy truncated stubs into lost output channel
      for (int region = 0; region < setup->numOverlappingRegions(); region++)
        copy_if(
            limit, stubs.end(), back_inserter(lost_[region]), [region](Stub* stub) { return stub->inRegion(region); });
      // remove truncated stubs form input channel
      stubs.erase(limit, stubs.end());
    }
  }

  // board level routing in two steps and products filling
  void DTC::produce(TTDTC& productAccepted, TTDTC& productLost) {
    // router step 1: merges stubs of all modules connected to one routing block into one stream
    Stubs lost;
    Stubss blockStubs(setup_->dtcNumRoutingBlocks());
    for (int routingBlock = 0; routingBlock < setup_->dtcNumRoutingBlocks(); routingBlock++)
      merge(input_[routingBlock], blockStubs[routingBlock], lost);
    // copy lost stubs during merge into lost output channel
    for (int region = 0; region < setup_->numOverlappingRegions(); region++) {
      auto inRegion = [region](Stub* stub) { return stub->inRegion(region); };
      copy_if(lost.begin(), lost.end(), back_inserter(lost_[region]), inRegion);
    }
    // router step 2: merges stubs of all routing blocks and splits stubs into one stream per overlapping region
    Stubss regionStubs(setup_->numOverlappingRegions());
    split(blockStubs, regionStubs);
    // fill products
    produce(regionStubs, productAccepted);
    produce(lost_, productLost);
  }

  // router step 1: merges stubs of all modules connected to one routing block into one stream
  void DTC::merge(Stubss& inputs, Stubs& output, Stubs& lost) {
    // for each input one fifo
    Stubss stacks(inputs.size());
    // clock accurate firmware emulation, each while trip describes one clock tick
    while (!all_of(inputs.begin(), inputs.end(), [](const Stubs& channel) { return channel.empty(); }) or
           !all_of(stacks.begin(), stacks.end(), [](const Stubs& channel) { return channel.empty(); })) {
      // fill fifos
      for (int iInput = 0; iInput < (int)inputs.size(); iInput++) {
        Stubs& input = inputs[iInput];
        Stubs& stack = stacks[iInput];
        if (input.empty())
          continue;
        Stub* stub = pop_front(input);
        if (stub) {
          if (enableTruncation_ && (int)stack.size() == setup_->dtcDepthMemory() - 1)
            // kill current first stub when fifo overflows
            lost.push_back(pop_front(stack));
          stack.push_back(stub);
        }
      }
      // route stub from a fifo to output if possible
      bool nothingToRoute(true);
      for (int iInput = inputs.size() - 1; iInput >= 0; iInput--) {
        Stubs& stack = stacks[iInput];
        if (stack.empty())
          continue;
        nothingToRoute = false;
        output.push_back(pop_front(stack));
        // only one stub can be routed to output per clock tick
        break;
      }
      // each clock tick output will grow by one, if no stub is available then by a gap
      if (nothingToRoute)
        output.push_back(nullptr);
    }
    // truncate if desired
    if (enableTruncation_ && (int)output.size() > setup_->numFramesIO()) {
      const auto limit = next(output.begin(), setup_->numFramesIO());
      copy_if(limit, output.end(), back_inserter(lost), [](Stub* stub) { return stub; });
      output.erase(limit, output.end());
    }
    // remove all gaps between end and last stub
    for (auto it = output.end(); it != output.begin();)
      it = (*--it) ? output.begin() : output.erase(it);
  }

  // router step 2: merges stubs of all routing blocks and splits stubs into one stream per overlapping region
  void DTC::split(Stubss& inputs, Stubss& outputs) {
    int region(0);
    auto regionMask = [&region](Stub* stub) { return stub && stub->inRegion(region) ? stub : nullptr; };
    for (Stubs& output : outputs) {
      // copy of masked inputs for each output
      Stubss streams(inputs.size());
      int i(0);
      for (Stubs& input : inputs) {
        Stubs& stream = streams[i++];
        transform(input.begin(), input.end(), back_inserter(stream), regionMask);
        for (auto it = stream.end(); it != stream.begin();)
          it = (*--it) ? stream.begin() : stream.erase(it);
      }
      merge(streams, output, lost_[region++]);
    }
  }

  // conversion from Stubss to TTDTC
  void DTC::produce(const Stubss& stubss, TTDTC& product) {
    int channel(0);
    auto toFrame = [&channel](Stub* stub) {
      return stub ? make_pair(stub->ttStubRef(), stub->frame(channel)) : FrameStub();
    };
    for (const Stubs& stubs : stubss) {
      StreamStub stream;
      stream.reserve(stubs.size());
      transform(stubs.begin(), stubs.end(), back_inserter(stream), toFrame);
      product.setStream(region_, board_, channel++, stream);
    }
  }

  // pop_front function which additionally returns copy of deleted front
  Stub* DTC::pop_front(Stubs& deque) {
    Stub* stub = deque.front();
    deque.pop_front();
    return stub;
  }

}  // namespace trackerDTC