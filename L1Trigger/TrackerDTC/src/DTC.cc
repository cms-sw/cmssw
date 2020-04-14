#include "L1Trigger/TrackerDTC/interface/DTC.h"
#include "L1Trigger/TrackerDTC/interface/Settings.h"
#include "L1Trigger/TrackerDTC/interface/Module.h"

#include <vector>
#include <deque>

using namespace std;
using namespace edm;

namespace trackerDTC {

  DTC::DTC(Settings* settings, int nStubs) : settings_(settings) { stubs_.reserve(nStubs); }

  // convert and assign TTStubRef to DTC routing block channel
  void DTC::consume(const vector<TTStubRef>& ttStubRefStream, Module* module) {
    for (const TTStubRef& ttStubRef : ttStubRefStream)
      stubs_.emplace_back(settings_, module, ttStubRef);
  }

  // board level routing in two steps and product filling
  void DTC::produce(TTDTC& product, int dtcId) {
    stubs_.shrink_to_fit();
    // outer tracker detector region [0-8]
    const int region = dtcId / settings_->numDTCsPerRegion();
    // outer tracker dtc id in region [0-23]
    const int board = dtcId % settings_->numDTCsPerRegion();
    // empty input, intermediate and output container
    Stubsss moduleStubs(settings_->numRoutingBlocks(), Stubss(settings_->numModulesPerRoutingBlock()));
    Stubss blockStubs(settings_->numRoutingBlocks());
    Stubss regionStubs(settings_->numOverlappingRegions());

    // fill input
    for (Stub& stub : stubs_)
      if (stub.valid())  // pt and eta cut
        moduleStubs[stub.blockId()][stub.channelId()].push_back(&stub);

    // sort stubs by bend
    for (auto& block : moduleStubs)
      for (auto& channel : block)
        sort(channel.begin(), channel.end(), [](Stub* lhs, Stub* rhs) { return abs(lhs->bend()) < abs(rhs->bend()); });

    // router step 1: merges stubs of all modules connected to one routing block into one stream
    for (int routingBlock = 0; routingBlock < settings_->numRoutingBlocks(); routingBlock++)
      merge(moduleStubs[routingBlock], blockStubs[routingBlock]);

    // router step 2: merges stubs of all routing blocks and splits stubs into one stream per overlapping region
    split(blockStubs, regionStubs);

    // fill product
    for (int channel = 0; channel < settings_->numOverlappingRegions(); channel++) {
      Stubs& stubs = regionStubs[channel];
      // truncate if desired
      if (settings_->enableTruncation())
        stubs.resize(min((int)stubs.size(), settings_->maxFramesChannelOutput()));
      // remove all gaps between end and last stub
      for (auto it = stubs.end(); it != stubs.begin();)
        it = (*--it) ? stubs.begin() : stubs.erase(it);
      // convert to TTDTC::Stream
      TTDTC::Stream stream;
      stream.reserve(stubs.size());
      for (const Stub* stub : stubs) {
        if (stub)
          stream.emplace_back(stub->ttStubRef(), stub->frame(channel));
        else
          // use default constructed TTDTC::Pair to represent gaps
          stream.emplace_back();
      }
      product.setStream(region, board, channel, stream);
    }
  }

  // router step 1: merges stubs of all modules connected to one routing block into one stream
  void DTC::merge(Stubss& inputs, Stubs& output) {
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
          if (settings_->enableTruncation() && (int)stack.size() == settings_->sizeStack() - 1)
            // kill current first stub when fifo overflows
            stack.pop_front();
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
  }

  // router step 2: merges stubs of all routing blocks and splits stubs into one stream per overlapping region
  void DTC::split(Stubss& inputs, Stubss& outputs) {
    int region(0);
    auto regionMask = [region](Stub* stub) { return stub->inRegion(region) ? stub : nullptr; };
    for (Stubs& output : outputs) {
      // copy of masked inputs for each output
      Stubss streams(inputs.size());
      int i(0);
      for (Stubs& input : inputs)
        transform(input.begin(), input.end(), back_inserter(streams[i++]), regionMask);
      merge(streams, output);
      region++;
    }
  }

  // pop_front function which additionally returns copy of deleted front
  Stub* DTC::pop_front(Stubs& deque) {
    Stub* stub = deque.front();
    deque.pop_front();
    return stub;
  }

}  // namespace trackerDTC