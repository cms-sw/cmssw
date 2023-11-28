#include "L1Trigger/TrackerTFP/interface/HoughTransform.h"

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

  HoughTransform::HoughTransform(const ParameterSet& iConfig,
                                 const Setup* setup,
                                 const DataFormats* dataFormats,
                                 int region)
      : enableTruncation_(iConfig.getParameter<bool>("EnableTruncation")),
        setup_(setup),
        dataFormats_(dataFormats),
        inv2R_(dataFormats_->format(Variable::inv2R, Process::ht)),
        phiT_(dataFormats_->format(Variable::phiT, Process::ht)),
        region_(region),
        input_(dataFormats_->numChannel(Process::ht), vector<deque<StubGP*>>(dataFormats_->numChannel(Process::gp))) {}

  // read in and organize input product (fill vector input_)
  void HoughTransform::consume(const StreamsStub& streams) {
    const int offset = region_ * dataFormats_->numChannel(Process::gp);
    auto validFrame = [](int sum, const FrameStub& frame) { return sum + (frame.first.isNonnull() ? 1 : 0); };
    int nStubsGP(0);
    for (int sector = 0; sector < dataFormats_->numChannel(Process::gp); sector++) {
      const StreamStub& stream = streams[offset + sector];
      nStubsGP += accumulate(stream.begin(), stream.end(), 0, validFrame);
    }
    stubsGP_.reserve(nStubsGP);
    for (int sector = 0; sector < dataFormats_->numChannel(Process::gp); sector++) {
      const int sectorPhi = sector % setup_->numSectorsPhi();
      const int sectorEta = sector / setup_->numSectorsPhi();
      for (const FrameStub& frame : streams[offset + sector]) {
        // Store input stubs in vector, so rest of HT algo can work with pointers to them (saves CPU)
        StubGP* stub = nullptr;
        if (frame.first.isNonnull()) {
          stubsGP_.emplace_back(frame, dataFormats_, sectorPhi, sectorEta);
          stub = &stubsGP_.back();
        }
        for (int binInv2R = 0; binInv2R < dataFormats_->numChannel(Process::ht); binInv2R++)
          input_[binInv2R][sector].push_back(stub && stub->inInv2RBin(binInv2R) ? stub : nullptr);
      }
    }
    // remove all gaps between end and last stub
    for (vector<deque<StubGP*>>& input : input_)
      for (deque<StubGP*>& stubs : input)
        for (auto it = stubs.end(); it != stubs.begin();)
          it = (*--it) ? stubs.begin() : stubs.erase(it);
    auto validStub = [](int sum, StubGP* stub) { return sum + (stub ? 1 : 0); };
    int nStubsHT(0);
    for (const vector<deque<StubGP*>>& binInv2R : input_)
      for (const deque<StubGP*>& sector : binInv2R)
        nStubsHT += accumulate(sector.begin(), sector.end(), 0, validStub);
    stubsHT_.reserve(nStubsHT);
  }

  // fill output products
  void HoughTransform::produce(StreamsStub& accepted, StreamsStub& lost) {
    for (int binInv2R = 0; binInv2R < dataFormats_->numChannel(Process::ht); binInv2R++) {
      const int inv2R = inv2R_.toSigned(binInv2R);
      deque<StubHT*> acceptedAll;
      deque<StubHT*> lostAll;
      for (deque<StubGP*>& inputSector : input_[binInv2R]) {
        const int size = inputSector.size();
        vector<StubHT*> acceptedSector;
        vector<StubHT*> lostSector;
        acceptedSector.reserve(size);
        lostSector.reserve(size);
        // associate stubs with inv2R and phiT bins
        fillIn(inv2R, inputSector, acceptedSector, lostSector);
        // Process::ht collects all stubs before readout starts -> remove all gaps
        acceptedSector.erase(remove(acceptedSector.begin(), acceptedSector.end(), nullptr), acceptedSector.end());
        // identify tracks
        readOut(acceptedSector, lostSector, acceptedAll, lostAll);
      }
      // truncate accepted stream
      const auto limit = enableTruncation_
                             ? next(acceptedAll.begin(), min(setup_->numFrames(), (int)acceptedAll.size()))
                             : acceptedAll.end();
      copy_if(limit, acceptedAll.end(), back_inserter(lostAll), [](StubHT* stub) { return stub; });
      acceptedAll.erase(limit, acceptedAll.end());
      // store found tracks
      auto put = [](const deque<StubHT*>& stubs, StreamStub& stream) {
        stream.reserve(stubs.size());
        for (StubHT* stub : stubs)
          stream.emplace_back(stub ? stub->frame() : FrameStub());
      };
      const int offset = region_ * dataFormats_->numChannel(Process::ht);
      put(acceptedAll, accepted[offset + binInv2R]);
      // store lost tracks
      put(lostAll, lost[offset + binInv2R]);
    }
  }

  // associate stubs with phiT bins in this inv2R column
  void HoughTransform::fillIn(int inv2R,
                              deque<StubGP*>& inputSector,
                              vector<StubHT*>& acceptedSector,
                              vector<StubHT*>& lostSector) {
    // fifo, used to store stubs which belongs to a second possible track
    deque<StubHT*> stack;
    // clock accurate firmware emulation, each while trip describes one clock tick, one stub in and one stub out per tick
    while (!inputSector.empty() || !stack.empty()) {
      StubHT* stubHT = nullptr;
      StubGP* stubGP = pop_front(inputSector);
      if (stubGP) {
        const double phiT = stubGP->phi() - inv2R_.floating(inv2R) * stubGP->r();
        const int major = phiT_.integer(phiT);
        if (phiT_.inRange(major)) {
          // major candidate has pt > threshold (3 GeV)
          // stubHT records which HT bin this stub is added to
          stubsHT_.emplace_back(*stubGP, major, inv2R);
          stubHT = &stubsHT_.back();
        }
        const double chi = phiT - phiT_.floating(major);
        if (abs(stubGP->r() * inv2R_.base()) + 2. * abs(chi) >= phiT_.base()) {
          // stub belongs to two candidates
          const int minor = chi >= 0. ? major + 1 : major - 1;
          if (phiT_.inRange(minor)) {
            // second (minor) candidate has pt > threshold (3 GeV)
            stubsHT_.emplace_back(*stubGP, minor, inv2R);
            if (enableTruncation_ && (int)stack.size() == setup_->htDepthMemory() - 1)
              // buffer overflow
              lostSector.push_back(pop_front(stack));
            // store minor stub in fifo
            stack.push_back(&stubsHT_.back());
          }
        }
      }
      // take a minor stub if no major stub available
      acceptedSector.push_back(stubHT ? stubHT : pop_front(stack));
    }
    // truncate to many input stubs
    const auto limit = enableTruncation_
                           ? next(acceptedSector.begin(), min(setup_->numFrames(), (int)acceptedSector.size()))
                           : acceptedSector.end();
    copy_if(limit, acceptedSector.end(), back_inserter(lostSector), [](StubHT* stub) { return stub; });
    acceptedSector.erase(limit, acceptedSector.end());
  }

  // identify tracks
  void HoughTransform::readOut(const vector<StubHT*>& acceptedSector,
                               const vector<StubHT*>& lostSector,
                               deque<StubHT*>& acceptedAll,
                               deque<StubHT*>& lostAll) const {
    // used to recognise in which order tracks are found
    TTBV trkFoundPhiTs(0, setup_->htNumBinsPhiT());
    // hitPattern for all possible tracks, used to find tracks
    vector<TTBV> patternHits(setup_->htNumBinsPhiT(), TTBV(0, setup_->numLayers()));
    // found unsigned phiTs, ordered in time
    vector<int> binsPhiT;
    // stub container for all possible tracks
    vector<vector<StubHT*>> tracks(setup_->htNumBinsPhiT());
    for (int binPhiT = 0; binPhiT < setup_->htNumBinsPhiT(); binPhiT++) {
      const int phiT = phiT_.toSigned(binPhiT);
      auto samePhiT = [phiT](int sum, StubHT* stub) { return sum + (stub->phiT() == phiT); };
      const int numAccepted = accumulate(acceptedSector.begin(), acceptedSector.end(), 0, samePhiT);
      const int numLost = accumulate(lostSector.begin(), lostSector.end(), 0, samePhiT);
      tracks[binPhiT].reserve(numAccepted + numLost);
    }
    for (StubHT* stub : acceptedSector) {
      const int binPhiT = phiT_.toUnsigned(stub->phiT());
      TTBV& pattern = patternHits[binPhiT];
      pattern.set(stub->layer());
      tracks[binPhiT].push_back(stub);
      if (pattern.count() >= setup_->htMinLayers() && !trkFoundPhiTs[binPhiT]) {
        // first time track found
        trkFoundPhiTs.set(binPhiT);
        binsPhiT.push_back(binPhiT);
      }
    }
    // read out found tracks ordered as found
    for (int binPhiT : binsPhiT) {
      const vector<StubHT*>& track = tracks[binPhiT];
      acceptedAll.insert(acceptedAll.end(), track.begin(), track.end());
    }
    // look for lost tracks
    for (StubHT* stub : lostSector) {
      const int binPhiT = phiT_.toUnsigned(stub->phiT());
      if (!trkFoundPhiTs[binPhiT])
        tracks[binPhiT].push_back(stub);
    }
    for (int binPhiT : trkFoundPhiTs.ids(false)) {
      const vector<StubHT*>& track = tracks[binPhiT];
      set<int> layers;
      auto toLayer = [](StubHT* stub) { return stub->layer(); };
      transform(track.begin(), track.end(), inserter(layers, layers.begin()), toLayer);
      if ((int)layers.size() >= setup_->htMinLayers())
        lostAll.insert(lostAll.end(), track.begin(), track.end());
    }
  }

  // remove and return first element of deque, returns nullptr if empty
  template <class T>
  T* HoughTransform::pop_front(deque<T*>& ts) const {
    T* t = nullptr;
    if (!ts.empty()) {
      t = ts.front();
      ts.pop_front();
    }
    return t;
  }

}  // namespace trackerTFP
