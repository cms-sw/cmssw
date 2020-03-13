#include "L1Trigger/TrackerDTC/interface/DTC.h"
#include "L1Trigger/TrackerDTC/interface/Settings.h"
#include "L1Trigger/TrackerDTC/interface/Module.h"

#include <vector>
#include <deque>

using namespace std;
using namespace edm;

namespace TrackerDTC {

  DTC::DTC(Settings* settings, const int& dtcId, const std::vector<Module*>& modules, const int& nStubs)
      : settings_(settings),                             // helper class to store configurations
        region_(dtcId / settings_->numDTCsPerRegion()),  // outer tracker detector region [0-8]
        board_(dtcId % settings_->numDTCsPerRegion()),   // outer tracker dtc id in region [0-23]
        modules_(modules)                                // container of sensor modules connected to this DTC
  {
    stubs_.reserve(nStubs);  // container of stubs on this DTC
  }

  // convert and assign TTStubRef to DTC routing block channel
  void DTC::consume(const vector<TTStubRef>& ttStubRefStream, const int& channelId) {
    for (const TTStubRef& ttStubRef : ttStubRefStream)
      stubs_.emplace_back(settings_, ttStubRef, modules_[channelId]);  // convert TTStub
  }

  // board level routing in two steps and product filling
  void DTC::produce(TTDTC& product) {
    // empty input container
    Stubsss moduleStubs(settings_->numRoutingBlocks(), Stubss(settings_->numModulesPerRoutingBlock()));
    Stubss blockStubs(settings_->numRoutingBlocks());        // empty intermediate container
    Stubss regionStubs(settings_->numOverlappingRegions());  // empty output container

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

      if (settings_->enableTruncation())  // truncate if desired
        stubs.resize(min((int)stubs.size(), settings_->maxFramesChannelOutput()));

      // remove all gaps between end and last stub
      for (auto it = stubs.end(); it != stubs.begin();)
        it = (*--it) ? stubs.begin() : stubs.erase(it);

      TTDTC::Stream stream;
      stream.reserve(stubs.size());
      for (const Stub* stub : stubs) {
        if (stub)
          stream.emplace_back(stub->ttStubRef(), stub->frame(channel));
        else  // use default constructed TTDTC::Pair to represent gaps
          stream.emplace_back();
      }
      product.setStream(region_, board_, channel, stream);
    }
  }

  // router step 1: merges stubs of all modules connected to one routing block into one stream
  void DTC::merge(Stubss& inputs, Stubs& output) {
    Stubss stacks(inputs.size());  // for each input one fifo

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
            stack.pop_front();  // kill current first stub when fifo overflows
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

        break;  // only one stub can be routed to output per clock tick
      }

      if (nothingToRoute)  // each clock tick output will grow by one, if no stub is available then by a gap
        output.push_back(nullptr);
    }
  }

  // router step 2: merges stubs of all routing blocks and splits stubs into one stream per overlapping region
  void DTC::split(Stubss& inputs, Stubss& outputs) {
    int region(0);
    for (Stubs& output : outputs) {
      Stubss streams(inputs.size());  // copy of inputs for each output

      int i(0);
      for (Stubs& input : inputs)
        transform(input.begin(), input.end(), back_inserter(streams[i++]), [region](Stub* stub) {
          return stub->inRegion(region) ? stub : nullptr;
        });

      merge(streams, output);

      region++;
    }
  }

  // new pop_front function which additionally returns copy of deleted front
  Stub* DTC::pop_front(Stubs& deque) {
    Stub* stub = deque.front();
    deque.pop_front();
    return stub;
  }

}  // namespace TrackerDTC