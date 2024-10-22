#include "L1Trigger/TrackerTFP/interface/MiniHoughTransform.h"

#include <numeric>
#include <algorithm>
#include <iterator>
#include <deque>
#include <vector>
#include <set>
#include <utility>
#include <cmath>

using namespace std;
using namespace edm;
using namespace tt;

namespace trackerTFP {

  MiniHoughTransform::MiniHoughTransform(const ParameterSet& iConfig,
                                         const Setup* setup,
                                         const DataFormats* dataFormats,
                                         int region)
      : enableTruncation_(iConfig.getParameter<bool>("EnableTruncation")),
        setup_(setup),
        dataFormats_(dataFormats),
        inv2R_(dataFormats_->format(Variable::inv2R, Process::ht)),
        phiT_(dataFormats_->format(Variable::phiT, Process::ht)),
        region_(region),
        numBinsInv2R_(setup_->htNumBinsInv2R()),
        numCells_(setup_->mhtNumCells()),
        numNodes_(setup_->mhtNumDLBNodes()),
        numChannel_(setup_->mhtNumDLBChannel()),
        input_(numBinsInv2R_) {}

  // read in and organize input product (fill vector input_)
  void MiniHoughTransform::consume(const StreamsStub& streams) {
    auto valid = [](int sum, const FrameStub& frame) { return sum + (frame.first.isNonnull() ? 1 : 0); };
    int nStubsHT(0);
    for (int binInv2R = 0; binInv2R < numBinsInv2R_; binInv2R++) {
      const StreamStub& stream = streams[region_ * numBinsInv2R_ + binInv2R];
      nStubsHT += accumulate(stream.begin(), stream.end(), 0, valid);
    }
    stubsHT_.reserve(nStubsHT);
    stubsMHT_.reserve(nStubsHT * numCells_);
    for (int binInv2R = 0; binInv2R < numBinsInv2R_; binInv2R++) {
      const int inv2R = inv2R_.toSigned(binInv2R);
      const StreamStub& stream = streams[region_ * numBinsInv2R_ + binInv2R];
      vector<StubHT*>& stubs = input_[binInv2R];
      stubs.reserve(stream.size());
      // Store input stubs in vector, so rest of MHT algo can work with pointers to them (saves CPU)
      for (const FrameStub& frame : stream) {
        StubHT* stub = nullptr;
        if (frame.first.isNonnull()) {
          stubsHT_.emplace_back(frame, dataFormats_, inv2R);
          stub = &stubsHT_.back();
        }
        stubs.push_back(stub);
      }
    }
  }

  // fill output products
  void MiniHoughTransform::produce(StreamsStub& accepted, StreamsStub& lost) {
    // fill MHT cells
    vector<deque<StubMHT*>> stubsCells(numBinsInv2R_ * numCells_);
    for (int channel = 0; channel < numBinsInv2R_; channel++)
      fill(channel, input_[channel], stubsCells);
    // perform static load balancing
    vector<vector<StubMHT*>> streamsSLB(numBinsInv2R_);
    for (int channel = 0; channel < numBinsInv2R_; channel++) {
      vector<deque<StubMHT*>> tmp(numCells_);
      // gather streams to mux together: same MHT cell of 4 adjacent MHT input streams
      for (int k = 0; k < numCells_; k++)
        swap(tmp[k], stubsCells[(channel / numCells_) * numBinsInv2R_ + channel % numCells_ + k * numCells_]);
      slb(tmp, streamsSLB[channel], lost[channel]);
    }
    // dynamic load balancing stage 1
    vector<vector<StubMHT*>> streamsDLB(numBinsInv2R_);
    for (int node = 0; node < numNodes_; node++) {
      vector<vector<StubMHT*>> tmp(numChannel_);
      // gather streams to dynamically balance them
      for (int k = 0; k < numChannel_; k++)
        swap(tmp[k], streamsSLB[(node / numCells_) * numNodes_ + node % numCells_ + k * numCells_]);
      dlb(tmp);
      for (int k = 0; k < numChannel_; k++)
        swap(tmp[k], streamsDLB[node * numChannel_ + k]);
    }
    // dynamic load balancing stage 2
    vector<vector<StubMHT*>> streamsMHT(numBinsInv2R_);
    for (int node = 0; node < numNodes_; node++) {
      vector<vector<StubMHT*>> tmp(numChannel_);
      // gather streams to dynamically balance them
      for (int k = 0; k < numChannel_; k++)
        swap(tmp[k], streamsDLB[node + k * numNodes_]);
      dlb(tmp);
      for (int k = 0; k < numChannel_; k++)
        swap(tmp[k], streamsMHT[node * numChannel_ + k]);
    }
    // fill output product
    for (int channel = 0; channel < numBinsInv2R_; channel++) {
      const vector<StubMHT*>& stubs = streamsMHT[channel];
      StreamStub& stream = accepted[region_ * numBinsInv2R_ + channel];
      stream.reserve(stubs.size());
      for (StubMHT* stub : stubs)
        stream.emplace_back(stub ? stub->frame() : FrameStub());
    }
  }

  // perform finer pattern recognition per track
  void MiniHoughTransform::fill(int channel, const vector<StubHT*>& stubs, vector<deque<StubMHT*>>& streams) {
    if (stubs.empty())
      return;
    int id;
    auto differentHT = [&id](StubHT* stub) { return id != stub->trackId(); };
    auto differentMHT = [&id](StubMHT* stub) { return !stub || id != stub->trackId(); };
    for (auto it = stubs.begin(); it != stubs.end();) {
      const auto start = it;
      id = (*it)->trackId();
      it = find_if(it, stubs.end(), differentHT);
      const int size = distance(start, it);
      // create finer track candidates stub container
      vector<vector<StubMHT*>> mhtCells(numCells_);
      for (vector<StubMHT*>& mhtCell : mhtCells)
        mhtCell.reserve(size);
      // fill finer track candidates stub container
      for (auto stub = start; stub != it; stub++) {
        const double r = (*stub)->r();
        const double chi = (*stub)->phi();
        // identify finer track candidates for this stub
        // 0 and 1 belong to the MHT cells with larger inv2R; 0 and 2 belong to those with smaller track PhiT
        vector<int> cells;
        cells.reserve(numCells_);
        const bool compA = 2. * abs(chi) < phiT_.base();
        const bool compB = 2. * abs(chi) < abs(r * inv2R_.base());
        const bool compAB = compA && compB;
        if (chi >= 0. && r >= 0.) {
          cells.push_back(3);
          if (compA)
            cells.push_back(1);
          if (compAB)
            cells.push_back(2);
        }
        if (chi >= 0. && r < 0.) {
          cells.push_back(1);
          if (compA)
            cells.push_back(3);
          if (compAB)
            cells.push_back(0);
        }
        if (chi < 0. && r >= 0.) {
          cells.push_back(0);
          if (compA)
            cells.push_back(2);
          if (compAB)
            cells.push_back(1);
        }
        if (chi < 0. && r < 0.) {
          cells.push_back(2);
          if (compA)
            cells.push_back(0);
          if (compAB)
            cells.push_back(3);
        }
        // organise stubs in finer track candidates
        for (int cell : cells) {
          const int inv2R = cell / setup_->mhtNumBinsPhiT();
          const int phiT = cell % setup_->mhtNumBinsPhiT();
          stubsMHT_.emplace_back(**stub, phiT, inv2R);
          mhtCells[cell].push_back(&stubsMHT_.back());
        }
      }
      // perform pattern recognition
      for (int sel = 0; sel < numCells_; sel++) {
        deque<StubMHT*>& stream = streams[channel * numCells_ + sel];
        vector<StubMHT*>& mhtCell = mhtCells[sel];
        set<int> layers;
        auto toLayer = [](StubMHT* stub) { return stub->layer(); };
        transform(mhtCell.begin(), mhtCell.end(), inserter(layers, layers.begin()), toLayer);
        if ((int)layers.size() < setup_->mhtMinLayers())
          mhtCell.clear();
        for (StubMHT* stub : mhtCell)
          stream.push_back(stub);
        stream.insert(stream.end(), size - (int)mhtCell.size(), nullptr);
      }
    }
    for (int sel = 0; sel < numCells_; sel++) {
      deque<StubMHT*>& stream = streams[channel * numCells_ + sel];
      // remove all gaps between end and last stub
      for (auto it = stream.end(); it != stream.begin();)
        it = (*--it) ? stream.begin() : stream.erase(it);
      // read out fine track cannot start before rough track has read in completely, add gaps to take this into account
      int pos(0);
      for (auto it = stream.begin(); it != stream.end();) {
        if (!(*it)) {
          it = stream.erase(it);
          continue;
        }
        id = (*it)->trackId();
        const int s = distance(it, find_if(it, stream.end(), differentMHT));
        const int d = distance(stream.begin(), it);
        pos += s;
        if (d < pos) {
          const int diff = pos - d;
          it = stream.insert(it, diff, nullptr);
          it = next(it, diff);
        } else
          it = stream.erase(remove(next(stream.begin(), pos), it, nullptr), it);
        it = next(it, s);
      }
      // adjust stream start so that first output stub is in first place in case of quickest track
      if (!stream.empty())
        stream.erase(stream.begin(), next(stream.begin(), setup_->mhtMinLayers()));
    }
  }

  // Static load balancing of inputs: mux 4 streams to 1 stream
  void MiniHoughTransform::slb(vector<deque<StubMHT*>>& inputs, vector<StubMHT*>& accepted, StreamStub& lost) const {
    if (all_of(inputs.begin(), inputs.end(), [](const deque<StubMHT*>& stubs) { return stubs.empty(); }))
      return;
    auto size = [](int sum, const deque<StubMHT*>& stubs) { return sum = stubs.size(); };
    const int nFrames = accumulate(inputs.begin(), inputs.end(), 0, size);
    accepted.reserve(nFrames);
    // input fifos
    vector<deque<StubMHT*>> stacks(numCells_);
    // helper for handshake
    TTBV empty(-1, numCells_, true);
    TTBV enable(0, numCells_);
    // clock accurate firmware emulation, each while trip describes one clock tick, one stub in and one stub out per tick
    while (!all_of(inputs.begin(), inputs.end(), [](const deque<StubMHT*>& d) { return d.empty(); }) or
           !all_of(stacks.begin(), stacks.end(), [](const deque<StubMHT*>& d) { return d.empty(); })) {
      // store stub in fifo
      for (int channel = 0; channel < numCells_; channel++) {
        StubMHT* stub = pop_front(inputs[channel]);
        if (stub)
          stacks[channel].push_back(stub);
      }
      // identify empty fifos
      for (int channel = 0; channel < numCells_; channel++)
        empty[channel] = stacks[channel].empty();
      // chose new fifo to read from if current fifo got empty
      const int iEnableOld = enable.plEncode();
      if (enable.none() || empty[iEnableOld]) {
        enable.reset();
        const int iNotEmpty = empty.plEncode(false);
        if (iNotEmpty < numCells_)
          enable.set(iNotEmpty);
      }
      // read from chosen fifo
      const int iEnable = enable.plEncode();
      if (enable.any())
        accepted.push_back(pop_front(stacks[iEnable]));
      else
        // gap if no fifo has been chosen
        accepted.push_back(nullptr);
    }
    // perform truncation if desired
    if (enableTruncation_ && (int)accepted.size() > setup_->numFrames()) {
      const auto limit = next(accepted.begin(), setup_->numFrames());
      auto valid = [](int sum, StubMHT* stub) { return sum + (stub ? 1 : 0); };
      const int nLost = accumulate(limit, accepted.end(), 0, valid);
      lost.reserve(nLost);
      for (auto it = limit; it != accepted.end(); it++)
        if (*it)
          lost.emplace_back((*it)->frame());
      accepted.erase(limit, accepted.end());
    }
    // cosmetics -- remove gaps at the end of stream
    for (auto it = accepted.end(); it != accepted.begin();)
      it = (*--it) == nullptr ? accepted.erase(it) : accepted.begin();
  }

  // Dynamic load balancing of inputs: swapping parts of streams to balance the amount of tracks per stream
  void MiniHoughTransform::dlb(vector<vector<StubMHT*>>& streams) const {
    if (all_of(streams.begin(), streams.end(), [](const vector<StubMHT*>& stubs) { return stubs.empty(); }))
      return;
    auto maxSize = [](int size, const vector<StubMHT*>& stream) { return size = max(size, (int)stream.size()); };
    const int nMax = accumulate(streams.begin(), streams.end(), 0, maxSize);
    for (vector<StubMHT*>& stream : streams)
      stream.resize(nMax, nullptr);
    vector<int> prevTrks(numChannel_, -1);
    bool swapping(false);
    vector<int> loads(numChannel_, 0);
    for (int i = 0; i < nMax; i++) {
      TTBV newTrks(0, numChannel_);
      for (int k = 0; k < numChannel_; k++)
        if (!streams[numChannel_ - k - 1][i] && streams[k][i] && streams[k][i]->trackId() != prevTrks[k])
          newTrks.set(k);
      for (int k = 0; k < numChannel_; k++)
        if (newTrks[k])
          if ((swapping && loads[numChannel_ - k - 1] > loads[k]) ||
              (!swapping && loads[k] > loads[numChannel_ - k - 1]))
            swapping = !swapping;
      for (int k = 0; k < numChannel_; k++) {
        if (streams[k][i])
          loads[swapping ? numChannel_ - k - 1 : k]++;
        prevTrks[k] = streams[k][i] ? streams[k][i]->trackId() : -1;
      }
      if (swapping)
        swap(streams[0][i], streams[1][i]);
    }
    // remove all gaps between end and last stub
    for (vector<StubMHT*>& stream : streams)
      for (auto it = stream.end(); it != stream.begin();)
        it = (*--it) ? stream.begin() : stream.erase(it);
  }

  // remove and return first element of deque, returns nullptr if empty
  template <class T>
  T* MiniHoughTransform::pop_front(deque<T*>& ts) const {
    T* t = nullptr;
    if (!ts.empty()) {
      t = ts.front();
      ts.pop_front();
    }
    return t;
  }

}  // namespace trackerTFP
