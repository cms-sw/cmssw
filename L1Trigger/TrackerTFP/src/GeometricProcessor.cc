#include "L1Trigger/TrackerTFP/interface/GeometricProcessor.h"

#include <numeric>
#include <algorithm>
#include <iterator>
#include <deque>
#include <vector>

namespace trackerTFP {

  GeometricProcessor::GeometricProcessor(const tt::Setup* setup,
                                         const DataFormats* dataFormats,
                                         const LayerEncoding* layerEncoding,
                                         std::vector<StubGP>& stubs)
      : setup_(setup), dataFormats_(dataFormats), layerEncoding_(layerEncoding), stubs_(stubs) {
    numChannelIn_ = dataFormats_->numChannel(Process::pp);
    numChannelOut_ = dataFormats_->numChannel(Process::gp);
  }

  // fill output products
  void GeometricProcessor::produce(const std::vector<std::vector<StubPP*>>& streamsIn,
                                   std::vector<std::deque<StubGP*>>& streamsOut) {
    for (int channelOut = 0; channelOut < numChannelOut_; channelOut++) {
      // helper
      const int phiT = channelOut % setup_->gpNumBinsPhiT() - setup_->gpNumBinsPhiT() / 2;
      const int zT = channelOut / setup_->gpNumBinsPhiT() - setup_->gpNumBinsZT() / 2;
      auto valid = [phiT, zT](StubPP* stub) {
        const bool phiTValid = stub && phiT >= stub->phiTMin() && phiT <= stub->phiTMax();
        const bool zTValid = stub && zT >= stub->zTMin() && zT <= stub->zTMax();
        return (phiTValid && zTValid) ? stub : nullptr;
      };
      // input streams of stubs
      std::vector<std::deque<StubPP*>> inputs(numChannelIn_);
      for (int channelIn = 0; channelIn < numChannelIn_; channelIn++) {
        const std::vector<StubPP*>& streamIn = streamsIn[channelIn];
        std::transform(streamIn.begin(), streamIn.end(), std::back_inserter(inputs[channelIn]), valid);
      }
      // fifo for each stream
      std::vector<std::deque<StubGP*>> stacks(streamsIn.size());
      // output stream
      std::deque<StubGP*>& output = streamsOut[channelOut];
      // clock accurate firmware emulation, each while trip describes one clock tick, one stub in and one stub out per tick
      while (
          !std::all_of(inputs.begin(), inputs.end(), [](const std::deque<StubPP*>& stubs) { return stubs.empty(); }) ||
          !std::all_of(stacks.begin(), stacks.end(), [](const std::deque<StubGP*>& stubs) { return stubs.empty(); })) {
        // fill input fifos
        for (int channelIn = 0; channelIn < numChannelIn_; channelIn++) {
          std::deque<StubGP*>& stack = stacks[channelIn];
          StubPP* stub = pop_front(inputs[channelIn]);
          if (stub) {
            // convert stub
            StubGP* stubGP = produce(*stub, phiT, zT);
            // buffer overflow
            if (setup_->enableTruncation() && static_cast<int>(stack.size()) == setup_->gpDepthMemory() - 1)
              pop_front(stack);
            stack.push_back(stubGP);
          }
        }
        // merge input fifos to one stream, prioritizing higher input channel over lower channel
        bool nothingToRoute(true);
        for (int channelIn = 0; channelIn < numChannelIn_; channelIn++) {
          StubGP* stub = pop_front(stacks[channelIn]);
          if (stub) {
            nothingToRoute = false;
            output.push_back(stub);
            break;
          }
        }
        if (nothingToRoute)
          output.push_back(nullptr);
      }
      // truncate if desired
      if (setup_->enableTruncation() && static_cast<int>(output.size()) > setup_->numFramesHigh())
        output.resize(setup_->numFramesHigh());
      // remove all gaps between end and last stub
      for (auto it = output.end(); it != output.begin();)
        it = (*--it) ? output.begin() : output.erase(it);
    }
  }

  // convert stub
  StubGP* GeometricProcessor::produce(const StubPP& stub, int phiT, int zT) {
    const DataFormat& dfPhiT = dataFormats_->format(Variable::phiT, Process::gp);
    const DataFormat& dfZT = dataFormats_->format(Variable::zT, Process::gp);
    const DataFormat& dfCot = dataFormats_->format(Variable::cot, Process::gp);
    const DataFormat& dfR = dataFormats_->format(Variable::r, Process::gp);
    const DataFormat& dfL = dataFormats_->format(Variable::layer, Process::gp);
    const double cot = dfCot.digi(dfZT.floating(zT) / setup_->chosenRofZ());
    // determine kf layer id
    const std::vector<int>& le = layerEncoding_->layerEncoding(zT);
    const int layerId = setup_->layerId(stub.frame().first);
    const auto it = std::find(le.begin(), le.end(), layerId);
    const int kfLayerId = std::min(static_cast<int>(std::distance(le.begin(), it)), setup_->numLayers() - 1);
    // create data fields
    const double r = stub.r();
    const double phi = stub.phi() - dfPhiT.floating(phiT);
    const double z = stub.z() - (stub.r() + dfR.digi(setup_->chosenRofPhi())) * cot;
    TTBV layer(kfLayerId, dfL.width());
    if (stub.layer()[4]) {  // barrel
      layer.set(5);
      if (stub.layer()[3])  // psTilt
        layer.set(3);
      if (stub.layer().val(3) < 3)  // layerId < 3
        layer.set(4);
    } else if (stub.layer()[3])  // psTilt
      layer.set(4);
    const int inv2RMin = stub.inv2RMin();
    const int inv2RMax = stub.inv2RMax();
    stubs_.emplace_back(stub, r, phi, z, layer, inv2RMin, inv2RMax);
    return &stubs_.back();
  }

  // remove and return first element of deque, returns nullptr if empty
  template <class T>
  T* GeometricProcessor::pop_front(std::deque<T*>& ts) const {
    T* t = nullptr;
    if (!ts.empty()) {
      t = ts.front();
      ts.pop_front();
    }
    return t;
  }

}  // namespace trackerTFP
