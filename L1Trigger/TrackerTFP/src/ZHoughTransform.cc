#include "L1Trigger/TrackerTFP/interface/ZHoughTransform.h"

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

  ZHoughTransform::ZHoughTransform(const ParameterSet& iConfig,
                                   const Setup* setup,
                                   const DataFormats* dataFormats,
                                   int region)
      : enableTruncation_(iConfig.getParameter<bool>("EnableTruncation")),
        setup_(setup),
        dataFormats_(dataFormats),
        region_(region),
        input_(dataFormats->numChannel(Process::mht)),
        stage_(0) {}

  // read in and organize input product (fill vector input_)
  void ZHoughTransform::consume(const StreamsStub& streams) {
    auto valid = [](int sum, const FrameStub& frame) { return sum + (frame.first.isNonnull() ? 1 : 0); };
    const int offset = region_ * dataFormats_->numChannel(Process::mht);
    int nStubsMHT(0);
    for (int channel = 0; channel < dataFormats_->numChannel(Process::mht); channel++) {
      const StreamStub& stream = streams[offset + channel];
      nStubsMHT += accumulate(stream.begin(), stream.end(), 0, valid);
    }
    stubsZHT_.reserve(nStubsMHT * (setup_->zhtNumCells() * setup_->zhtNumStages()));
    for (int channel = 0; channel < dataFormats_->numChannel(Process::mht); channel++) {
      const StreamStub& stream = streams[offset + channel];
      vector<StubZHT*>& stubs = input_[channel];
      stubs.reserve(stream.size());
      // Store input stubs in vector, so rest of ZHT algo can work with pointers to them (saves CPU)
      for (const FrameStub& frame : stream) {
        StubZHT* stub = nullptr;
        if (frame.first.isNonnull()) {
          StubMHT stubMHT(frame, dataFormats_);
          stubsZHT_.emplace_back(stubMHT);
          stub = &stubsZHT_.back();
        }
        stubs.push_back(stub);
      }
    }
  }

  // fill output products
  void ZHoughTransform::produce(StreamsStub& accepted, StreamsStub& lost) {
    vector<deque<StubZHT*>> streams(dataFormats_->numChannel(Process::mht));
    for (int channel = 0; channel < dataFormats_->numChannel(Process::mht); channel++)
      streams[channel] = deque<StubZHT*>(input_[channel].begin(), input_[channel].end());
    vector<deque<StubZHT*>> stubsCells(dataFormats_->numChannel(Process::mht) * setup_->zhtNumCells());
    for (stage_ = 0; stage_ < setup_->zhtNumStages(); stage_++) {
      // fill ZHT cells
      for (int channel = 0; channel < dataFormats_->numChannel(Process::mht); channel++)
        fill(channel, streams[channel], stubsCells);
      // perform static load balancing
      for (int channel = 0; channel < dataFormats_->numChannel(Process::mht); channel++) {
        vector<deque<StubZHT*>> tmp(setup_->zhtNumCells());
        // gather streams to mux together: same ZHT cell of 4 adjacent ZHT input streams
        for (int k = 0; k < setup_->zhtNumCells(); k++)
          //swap(tmp[k], stubsCells[(channel / setup_->zhtNumCells()) * dataFormats_->numChannel(Process::mht) + channel % setup_->zhtNumCells() + k * setup_->zhtNumCells()]);
          swap(tmp[k], stubsCells[channel * setup_->zhtNumCells() + k]);
        slb(tmp, streams[channel], lost[channel]);
      }
    }
    // fill output product
    for (int channel = 0; channel < dataFormats_->numChannel(Process::mht); channel++) {
      deque<StubZHT*>& stubs = streams[channel];
      StreamStub& stream = accepted[region_ * dataFormats_->numChannel(Process::mht) + channel];
      merge(stubs, stream);
    }
  }

  // perform finer pattern recognition per track
  void ZHoughTransform::fill(int channel, const deque<StubZHT*>& stubs, vector<deque<StubZHT*>>& streams) {
    if (stubs.empty())
      return;
    const double baseZT =
        dataFormats_->format(Variable::zT, Process::zht).base() * pow(2, setup_->zhtNumStages() - stage_);
    const double baseCot =
        dataFormats_->format(Variable::cot, Process::zht).base() * pow(2, setup_->zhtNumStages() - stage_);
    int id;
    auto different = [&id](StubZHT* stub) { return !stub || id != stub->trackId(); };
    for (auto it = stubs.begin(); it != stubs.end();) {
      if (!*it) {
        const auto begin = find_if(it, stubs.end(), [](StubZHT* stub) { return stub; });
        const int nGaps = distance(it, begin);
        for (deque<StubZHT*>& stream : streams)
          stream.insert(stream.end(), nGaps, nullptr);
        it = begin;
        continue;
      }
      const auto start = it;
      const double cotGlobal = (*start)->cotf() + setup_->sectorCot((*start)->sectorEta());
      id = (*it)->trackId();
      it = find_if(it, stubs.end(), different);
      const int size = distance(start, it);
      // create finer track candidates stub container
      vector<vector<StubZHT*>> mhtCells(setup_->zhtNumCells());
      for (vector<StubZHT*>& mhtCell : mhtCells)
        mhtCell.reserve(size);
      // fill finer track candidates stub container
      for (auto stub = start; stub != it; stub++) {
        const double r = (*stub)->r() + setup_->chosenRofPhi() - setup_->chosenRofZ();
        const double chi = (*stub)->chi();
        const double dChi = setup_->dZ((*stub)->ttStubRef(), cotGlobal);
        // identify finer track candidates for this stub
        // 0 and 1 belong to the ZHT cells with smaller cot; 0 and 2 belong to those with smaller zT
        vector<int> cells;
        cells.reserve(setup_->zhtNumCells());
        const bool compA = 2. * abs(chi) < baseZT + dChi;
        const bool compB = 2. * abs(chi) < abs(r) * baseCot + dChi;
        const bool compC = 2. * abs(chi) < dChi;
        if (chi >= 0. && r >= 0.) {
          cells.push_back(1);
          if (compA)
            cells.push_back(3);
          if (compB)
            cells.push_back(0);
          if (compC)
            cells.push_back(2);
        }
        if (chi >= 0. && r < 0.) {
          cells.push_back(3);
          if (compA)
            cells.push_back(1);
          if (compB)
            cells.push_back(2);
          if (compC)
            cells.push_back(0);
        }
        if (chi < 0. && r >= 0.) {
          cells.push_back(2);
          if (compA)
            cells.push_back(0);
          if (compB)
            cells.push_back(3);
          if (compC)
            cells.push_back(1);
        }
        if (chi < 0. && r < 0.) {
          cells.push_back(0);
          if (compA)
            cells.push_back(2);
          if (compB)
            cells.push_back(1);
          if (compC)
            cells.push_back(3);
        }
        for (int cell : cells) {
          const double cot = (cell / setup_->zhtNumBinsZT() - .5) * baseCot / 2.;
          const double zT = (cell % setup_->zhtNumBinsZT() - .5) * baseZT / 2.;
          stubsZHT_.emplace_back(**stub, zT, cot, cell);
          mhtCells[cell].push_back(&stubsZHT_.back());
        }
      }
      // perform pattern recognition
      for (int sel = 0; sel < setup_->zhtNumCells(); sel++) {
        deque<StubZHT*>& stream = streams[channel * setup_->zhtNumCells() + sel];
        vector<StubZHT*>& mhtCell = mhtCells[sel];
        set<int> layers;
        auto toLayer = [](StubZHT* stub) { return stub->layer(); };
        transform(mhtCell.begin(), mhtCell.end(), inserter(layers, layers.begin()), toLayer);
        if ((int)layers.size() < setup_->mhtMinLayers())
          mhtCell.clear();
        for (StubZHT* stub : mhtCell)
          stream.push_back(stub);
        stream.insert(stream.end(), size - (int)mhtCell.size(), nullptr);
      }
    }
    for (int sel = 0; sel < setup_->zhtNumCells(); sel++) {
      deque<StubZHT*>& stream = streams[channel * setup_->zhtNumCells() + sel];
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
        const int s = distance(it, find_if(it, stream.end(), different));
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
  void ZHoughTransform::slb(vector<deque<StubZHT*>>& inputs, deque<StubZHT*>& accepted, StreamStub& lost) const {
    accepted.clear();
    if (all_of(inputs.begin(), inputs.end(), [](const deque<StubZHT*>& stubs) { return stubs.empty(); }))
      return;
    // input fifos
    vector<deque<StubZHT*>> stacks(setup_->zhtNumCells());
    // helper for handshake
    TTBV empty(-1, setup_->zhtNumCells(), true);
    TTBV enable(0, setup_->zhtNumCells());
    // clock accurate firmware emulation, each while trip describes one clock tick, one stub in and one stub out per tick
    while (!all_of(inputs.begin(), inputs.end(), [](const deque<StubZHT*>& d) { return d.empty(); }) or
           !all_of(stacks.begin(), stacks.end(), [](const deque<StubZHT*>& d) { return d.empty(); })) {
      // store stub in fifo
      for (int channel = 0; channel < setup_->zhtNumCells(); channel++) {
        StubZHT* stub = pop_front(inputs[channel]);
        if (stub)
          stacks[channel].push_back(stub);
      }
      // identify empty fifos
      for (int channel = 0; channel < setup_->zhtNumCells(); channel++)
        empty[channel] = stacks[channel].empty();
      // chose new fifo to read from if current fifo got empty
      const int iEnableOld = enable.plEncode();
      if (enable.none() || empty[iEnableOld]) {
        enable.reset();
        const int iNotEmpty = empty.plEncode(false);
        if (iNotEmpty < setup_->zhtNumCells())
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
      auto valid = [](int sum, StubZHT* stub) { return sum + (stub ? 1 : 0); };
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

  //
  void ZHoughTransform::merge(deque<StubZHT*>& stubs, StreamStub& stream) const {
    stubs.erase(remove(stubs.begin(), stubs.end(), nullptr), stubs.end());
    /*stream.reserve(stubs.size());
    transform(stubs.begin(), stubs.end(), back_inserter(stream), [](StubZHT* stub){ return stub->frame(); });
    return;*/
    map<int, set<pair<int, int>>> candidates;
    const int weight = setup_->zhtNumCells() * pow(2, setup_->zhtNumStages());
    for (const StubZHT* stub : stubs)
      candidates[stub->trackId() / weight].emplace(stub->cot(), stub->zT());
    vector<deque<FrameStub>> tracks(candidates.size());
    for (auto it = stubs.begin(); it != stubs.end();) {
      const auto start = it;
      const int id = (*it)->trackId();
      const int candId = id / weight;
      const auto m = candidates.find(candId);
      pair<int, int> cotp(9e9, -9e9);
      pair<int, int> zTp(9e9, -9e9);
      for (const pair<int, int>& para : m->second) {
        cotp = {min(cotp.first, para.first), max(cotp.second, para.first)};
        zTp = {min(zTp.first, para.second), max(zTp.second, para.second)};
      }
      const int cot = (cotp.first + cotp.second) / 2;
      const int zT = (cotp.first + cotp.second) / 2;
      const int pos = distance(candidates.begin(), m);
      deque<FrameStub>& track = tracks[pos];
      auto different = [id](const StubZHT* stub) { return id != stub->trackId(); };
      it = find_if(it, stubs.end(), different);
      for (auto s = start; s != it; s++) {
        if (find_if(track.begin(), track.end(), [s](const FrameStub& stub) {
              return (*s)->ttStubRef() == stub.first;
            }) != track.end())
          continue;
        const StubZHT stub(**s, cot, zT);
        track.push_back(stub.frame());
      }
    }
    const int size = accumulate(tracks.begin(), tracks.end(), 0, [](int sum, const deque<FrameStub>& stubs) {
      return sum + (int)stubs.size();
    });
    stream.reserve(size);
    for (deque<FrameStub>& track : tracks)
      for (const FrameStub& stub : track)
        stream.push_back(stub);
  }

  // remove and return first element of deque, returns nullptr if empty
  template <class T>
  T* ZHoughTransform::pop_front(deque<T*>& ts) const {
    T* t = nullptr;
    if (!ts.empty()) {
      t = ts.front();
      ts.pop_front();
    }
    return t;
  }

}  // namespace trackerTFP
