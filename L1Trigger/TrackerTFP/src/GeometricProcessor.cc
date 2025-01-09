#include "L1Trigger/TrackerTFP/interface/GeometricProcessor.h"

#include <numeric>
#include <algorithm>
#include <iterator>
#include <deque>
#include <vector>

using namespace std;
using namespace edm;
using namespace tt;

namespace trackerTFP {

  GeometricProcessor::GeometricProcessor(const ParameterSet& iConfig,
                                         const Setup* setup,
                                         const DataFormats* dataFormats,
                                         const LayerEncoding* layerEncoding,
                                         std::vector<StubGP>& stubs)
      : enableTruncation_(iConfig.getParameter<bool>("EnableTruncation")),
        setup_(setup),
        dataFormats_(dataFormats),
        layerEncoding_(layerEncoding),
        stubs_(stubs) {}

  // fill output products
  void GeometricProcessor::produce(const vector<vector<StubPP*>>& streamsIn, vector<deque<StubGP*>>& streamsOut) {
    static const int numChannelIn = dataFormats_->numChannel(Process::pp);
    static const int numChannelOut = dataFormats_->numChannel(Process::gp);
    for (int channelOut = 0; channelOut < numChannelOut; channelOut++) {
      // helper
      const int phiT = channelOut % setup_->gpNumBinsPhiT() - setup_->gpNumBinsPhiT() / 2;
      const int zT = channelOut / setup_->gpNumBinsPhiT() - setup_->gpNumBinsZT() / 2;
      auto valid = [phiT, zT](StubPP* stub) {
        const bool phiTValid = stub && phiT >= stub->phiTMin() && phiT <= stub->phiTMax();
        const bool zTValid = stub && zT >= stub->zTMin() && zT <= stub->zTMax();
        return (phiTValid && zTValid) ? stub : nullptr;
      };
      // input streams of stubs
      vector<deque<StubPP*>> inputs(numChannelIn);
      for (int channelIn = 0; channelIn < numChannelIn; channelIn++) {
        const vector<StubPP*>& streamIn = streamsIn[channelIn];
        transform(streamIn.begin(), streamIn.end(), back_inserter(inputs[channelIn]), valid);
      }
      // fifo for each stream
      vector<deque<StubGP*>> stacks(streamsIn.size());
      // output stream
      deque<StubGP*>& output = streamsOut[channelOut];
      // clock accurate firmware emulation, each while trip describes one clock tick, one stub in and one stub out per tick
      while (!all_of(inputs.begin(), inputs.end(), [](const deque<StubPP*>& stubs) { return stubs.empty(); }) or
             !all_of(stacks.begin(), stacks.end(), [](const deque<StubGP*>& stubs) { return stubs.empty(); })) {
        // fill input fifos
        for (int channelIn = 0; channelIn < numChannelIn; channelIn++) {
          deque<StubGP*>& stack = stacks[channelIn];
          StubPP* stub = pop_front(inputs[channelIn]);
          if (stub) {
            // convert stub
            StubGP* stubGP = produce(*stub, phiT, zT);
            // buffer overflow
            if (enableTruncation_ && (int)stack.size() == setup_->gpDepthMemory() - 1)
              pop_front(stack);
            stack.push_back(stubGP);
          }
        }
        // merge input fifos to one stream, prioritizing higher input channel over lower channel
        bool nothingToRoute(true);
        //for (int channelIn = numChannelIn - 1; channelIn >= 0; channelIn--) {
        for (int channelIn = 0; channelIn < numChannelIn; channelIn++) {
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
      if (enableTruncation_ && (int)output.size() > setup_->numFramesHigh())
        output.resize(setup_->numFramesHigh());
      // remove all gaps between end and last stub
      for (auto it = output.end(); it != output.begin();)
        it = (*--it) ? output.begin() : output.erase(it);
    }
  }

  // convert stub
  StubGP* GeometricProcessor::produce(const StubPP& stub, int phiT, int zT) {
    static const DataFormat& dfPhiT = dataFormats_->format(Variable::phiT, Process::gp);
    static const DataFormat& dfZT = dataFormats_->format(Variable::zT, Process::gp);
    static const DataFormat& dfCot = dataFormats_->format(Variable::cot, Process::gp);
    static const DataFormat& dfR = dataFormats_->format(Variable::r, Process::gp);
    static const DataFormat& dfL = dataFormats_->format(Variable::layer, Process::gp);
    const double cot = dfCot.digi(dfZT.floating(zT) / setup_->chosenRofZ());
    // determine kf layer id
    const vector<int>& le = layerEncoding_->layerEncoding(zT);
    const int layerId = setup_->layerId(stub.frame().first);
    const auto it = find(le.begin(), le.end(), layerId);
    const int kfLayerId = min((int)distance(le.begin(), it), setup_->numLayers() - 1);
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
  T* GeometricProcessor::pop_front(deque<T*>& ts) const {
    T* t = nullptr;
    if (!ts.empty()) {
      t = ts.front();
      ts.pop_front();
    }
    return t;
  }

}  // namespace trackerTFP
