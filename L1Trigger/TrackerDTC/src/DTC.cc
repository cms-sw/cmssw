#include "L1Trigger/TrackerDTC/interface/DTC.h"

#include <vector>
#include <sstream>
#include <cmath>
#include <iomanip>
#include <ostream>
#include <utility>
#include <algorithm>
#include <deque>
#include <fstream>
#include <cstdlib>
#include <iterator>

namespace trackerDTC {

  DTC::DTC(const Setup* setup,
           const Config& config,
           int dtcId,
           std::vector<TH1F*>& his,
           std::vector<TProfile*>& prof,
           TH2F* hisRZStubs,
           TH2F* hisRZLost)
      : setup_(setup),
        config_(&config),
        dtcId_(dtcId),
        his_(his),
        prof_(prof),
        hisRZStubs_(hisRZStubs),
        hisRZLost_(hisRZLost),
        stubsFE_(setup_->cicNumBX()),
        streamsFE_(setup_->dtcNumModule()),
        streamsIn_(setup_->dtcNumModule()),
        streamsOut_(setup_->regNumTFP() * setup_->sysNumOverlap()) {
    // prep input data container
    for (tt::Stream streamIn : streamsIn_)
      streamIn.reserve(config_->num8BX * (setup_->sysNumFramesInfra() + setup_->feNumFrames()));
    // prep output data container
    for (tt::Stream streamOut : streamsOut_)
      streamOut.reserve(config_->num18BX * (setup_->sysNumFramesInfra() + setup_->sysNumFrames()));
    // create header
    if (config_->enable) {
      auto header = [this](std::stringstream& ss, int nLinks) {
        static constexpr int nQuad = 4;
        // file header
        ss << "ID: CMSSW DTC " << dtcId_ << std::endl
           << "Metadata: (strobe,) start of orbit, start of packet, end of packet, valid" << std::endl
           << std::endl;
        // link header
        ss << "      Link  ";
        for (int link = 0; link < std::ceil(nLinks / nQuad) * nQuad; link++)
          ss << "            " << std::setfill('0') << std::setw(3) << link << "        ";
        ss << std::endl;
      };
      header(headerIn_, setup_->dtcNumModule());
      header(headerPre_, setup_->dtcNumTFP());
    }
    // prep paths
    pathIn_ = config_->pathIPBB + std::to_string(dtcId_) + "/in.txt";
    pathPre_ = config_->pathIPBB + std::to_string(dtcId_) + "/pre.txt";
    pathOut_ = config_->pathIPBB + std::to_string(dtcId_) + "/out.txt";
    pathDiff_ = config_->pathIPBB + std::to_string(dtcId_) + "/diff.txt";
    // prep system command: go to project area
    cmd_ << "cd " << config_->pathIPBB << dtcId_ << " && ";
    // run questasim and supress output
    cmd_ << "./run_sim -quiet -c work.top -do 'run " << config_->runTime << "us' -do 'quit' &> /dev/null && ";
    // compare output txt with predicted txt file
    cmd_ << "diff " << pathPre_ << " " << pathOut_ << " &> " << pathDiff_;
  }

  // process single bx
  void DTC::consume(const edm::Handle<TTStubDetSetVec>& handle, int bx) {
    std::vector<StubFE>& stubsFE = stubsFE_[bx];
    const std::vector<const SensorModule*>& sensorModules = setup_->dtcModules(dtcId_);
    std::vector<std::pair<const SensorModule*, TTStubDetSetVec::const_iterator>> modules;
    modules.reserve(setup_->dtcNumModule());
    int numStubs(0);
    // loop over all modules
    for (auto ttModule = handle->begin(); ttModule != handle->end(); ttModule++) {
      // find modules connected to dtc under test
      const DetId detId = ttModule->detId() + 1;
      // helper to find stubs for this DTC
      auto sameDetId = [&detId](const SensorModule* sm) { return sm && (sm->detId() == detId); };
      // check if this module is connected
      const auto it = std::find_if(sensorModules.begin(), sensorModules.end(), sameDetId);
      if (it == sensorModules.end())
        continue;
      // count stubs
      numStubs += ttModule->size();
      // store module
      modules.emplace_back(*it, ttModule);
    }
    // prep input data container
    stubsFE.reserve(numStubs);
    // loop over connected modules
    for (const auto& pair : modules) {
      std::deque<const StubFE*>& stream = streamsFE_[pair.first->modId()];
      // collect stubs connected to this DTC
      for (auto ttStub = pair.second->begin(); ttStub != pair.second->end(); ttStub++) {
        const TTStubRef ttStubRef(makeRefTo(handle, ttStub));
        stubsFE.emplace_back(setup_, pair.first, ttStubRef, bx);
        stream.push_back(&stubsFE.back());
        // fill histo
        const GlobalPoint gp = setup_->stubPosTT(ttStubRef);
        hisRZStubs_->Fill(gp.z(), gp.perp());
      }
    }
  }

  // process 8 bx boxcars
  void DTC::produce(int bx) {
    const int numOut8BX = setup_->cicNumBX() * setup_->sysNumOverlap();
    const int numOut18BX = setup_->regNumTFP() * setup_->sysNumOverlap();
    // create emulation input
    tt::Streams input(setup_->dtcNumModule());
    for (tt::Stream& stream : input)
      stream.reserve(setup_->feNumFrames());
    produce(input);
    // store input
    if (config_->enable) {
      for (int channel = 0; channel < setup_->dtcNumModule(); channel++) {
        const tt::Stream& boxcar = input[channel];
        tt::Stream& testFile = streamsIn_[channel];
        testFile.insert(testFile.end(), setup_->sysNumFramesInfra(), tt::Frame());
        testFile.insert(testFile.end(), boxcar.begin(), boxcar.end());
        testFile.insert(testFile.end(), setup_->feNumFrames() - boxcar.size(), tt::Frame());
      }
    }
    // emulate output
    tt::Streams output(numOut8BX);
    for (tt::Stream& stream : output)
      stream.reserve(setup_->sysNumFrames());
    produce(input, output);
    // store output
    if (config_->enable) {
      const int offset = (bx - setup_->cicNumBX()) * setup_->sysNumOverlap();
      for (int channel8BX = 0; channel8BX < numOut8BX; channel8BX++) {
        const int channel18BX = (offset + channel8BX) % numOut18BX;
        const tt::Stream& event = output[channel8BX];
        tt::Stream& testFile = streamsOut_[channel18BX];
        testFile.insert(testFile.end(), setup_->sysNumFramesInfra(), tt::Frame());
        testFile.insert(testFile.end(), event.begin(), event.end());
        testFile.insert(testFile.end(), setup_->sysNumFrames() - event.size(), tt::Frame());
      }
    }
    // prep for next round
    for (std::vector<StubFE>& stubs : stubsFE_)
      stubs.clear();
    for (std::deque<const StubFE*>& stream : streamsFE_)
      stream.clear();
  }

  // compare s/w with f/w
  void DTC::analyze() {
    if (config_->enable) {
      std::fstream fs;
      std::stringstream ss;
      auto convert = [&fs, &ss, this](const std::stringstream& header,
                                      const std::vector<tt::Stream>& streams,
                                      const std::string& path,
                                      int numFrames) {
        // file header
        ss << header.str();
        // write one line per frame for all channel
        for (int iFrame = 0; iFrame < static_cast<int>(streams.front().size()); iFrame++) {
          const int lFrame = iFrame % (numFrames + setup_->sysNumFramesInfra());
          // frame number
          ss << "Frame " << std::setfill('0') << std::setw(4) << std::dec << iFrame << "  ";
          for (const tt::Stream& stream : streams) {
            // channel frame header (start of orbit, start of packet, end of packet, valid)
            if (iFrame == setup_->sysNumFramesInfra())
              ss << "  1001 ";
            else if (lFrame < setup_->sysNumFramesInfra())
              ss << "  0000 ";
            else
              ss << "  0001 ";
            // channel frame data
            ss << std::setfill('0') << std::setw(TTBV::S_ / 4) << std::hex << stream[iFrame].to_ullong();
          }
          ss << std::endl;
        }
        // print
        fs.open(path, std::fstream::out);
        fs << ss.rdbuf();
        fs.close();
        ss.str("");
        ss.clear();
      };
      // convert input data to txt
      convert(headerIn_, streamsIn_, pathIn_, setup_->feNumFrames());
      // convert output data to txt
      convert(headerPre_, streamsOut_, pathPre_, setup_->sysNumFrames());
      // run modelsim and linux diff
      std::system(cmd_.str().c_str());
      // read diff output
      fs.open(pathDiff_, std::fstream::in);
      ss << fs.rdbuf();
      fs.close();
      // count lines, 4 are expected
      int n(0);
      std::string token;
      while (getline(ss, token))
        n++;
      if (n != 4)
        throw cms::Exception("BitError.") << "In DTC # " << dtcId_ << ".";
      else
        std::cout << "." << std::flush;
    }
    // prep for next round
    for (tt::Stream& stream : streamsIn_)
      stream.clear();
    for (tt::Stream& stream : streamsOut_)
      stream.clear();
  }

  // create emulation input
  void DTC::produce(tt::Streams& streams) {
    const std::vector<const SensorModule*>& dtcModules = setup_->dtcModules(dtcId_);
    for (int channel = 0; channel < setup_->dtcNumModule(); channel++) {
      const SensorModule* sm = dtcModules[channel];
      if (!sm)
        continue;
      const std::deque<const StubFE*>& streamFE = streamsFE_[channel];
      tt::Stream& stream = streams[channel];
      const bool gig10 = (dtcId_ % setup_->sysNumATCASlot()) < setup_->sysSlotLimitPS();
      const int max = gig10 ? setup_->cicNumStub10g() : setup_->cicNumStub5g();
      std::vector<std::deque<const StubFE*>> cics(setup_->smNumCIC());
      // loop over both CICs
      for (int iCIC = 0; iCIC < setup_->smNumCIC(); iCIC++) {
        std::deque<const StubFE*>& stubs = cics[iCIC];
        if (sm->psModule())
          // apply mpa rules, 4 stubs per 2 bx row priority, up to 2 stubs with same pos but different bend
          mpa(streamFE, stubs, iCIC);
        else
          // apply cbc rules, 3 stubs per bx row priority
          cbc(streamFE, stubs, iCIC);
        // apply cic rules, 35/16 stubs per 8 bx be-bend > row priority
        std::stable_sort(stubs.begin(), stubs.end(), [](const StubFE* lhs, const StubFE* rhs) {
          return std::abs(lhs->bend()) < std::abs(rhs->bend());
        });
        if (static_cast<int>(stubs.size()) > max)
          lost(stubs, max);
        stubs.resize(max, nullptr);
      }
      // convert stubs to frames
      for (int iFrame = 0; iFrame < max; iFrame++) {
        TTBV ttBV;
        for (int iCIC = 0; iCIC < setup_->smNumCIC(); iCIC++) {
          const StubFE* stub = cics[iCIC][iFrame];
          ttBV += stub ? stub->ttBV() : TTBV(0, TTBV::S_ / 2);
        }
        stream.push_back(ttBV.bs());
      }
    }
  }

  // apply mpa rules, 4 stubs per 2 bx row priority, up to 2 stubs with same pos but different bend
  void DTC::mpa(const std::deque<const StubFE*>& input, std::deque<const StubFE*>& output, int iCIC) {
    // loop over 4 times 2bx packet
    for (int bxIter = 0; bxIter < setup_->cicNumBX() / setup_->mpaNumBX(); bxIter++) {
      const int bxBegin = bxIter * setup_->mpaNumBX();
      const int bxEnd = bxBegin + setup_->mpaNumBX();
      // identify stub range for this 2bx packet
      const auto begin = std::find_if(input.begin(), input.end(), [bxBegin, bxEnd](const StubFE* s) {
        return s->bx() >= bxBegin && s->bx() < bxEnd;
      });
      const auto end = std::find_if(begin, input.end(), [bxEnd](const StubFE* s) { return s->bx() >= bxEnd; });
      // loop over 8 mpas connected to one cic
      for (int iMPA = 0; iMPA < setup_->cicNumFEC(); iMPA++) {
        // collect 2bx packet stubs for this mpa
        std::deque<const StubFE*> stubs;
        std::copy_if(begin, end, std::back_inserter(stubs), [iCIC, iMPA](const StubFE* s) {
          return s->cic() == iCIC && s->fec() == iMPA;
        });
        if (stubs.empty())
          continue;
        // check for only up to 2 stubs per row, col, bx combination, keep smallest abs(bend)
        auto equal = [](const StubFE* lhs, const StubFE* rhs) {
          return lhs->row() == rhs->row() && lhs->col() == rhs->col() && lhs->bx() == rhs->bx();
        };
        std::sort(stubs.begin(), stubs.end(), [](const StubFE* lhs, const StubFE* rhs) {
          return std::abs(lhs->bend()) < std::abs(rhs->bend());
        });
        std::stable_sort(
            stubs.begin(), stubs.end(), [](const StubFE* lhs, const StubFE* rhs) { return lhs->bx() < rhs->bx(); });
        std::stable_sort(
            stubs.begin(), stubs.end(), [](const StubFE* lhs, const StubFE* rhs) { return lhs->col() < rhs->col(); });
        std::stable_sort(
            stubs.begin(), stubs.end(), [](const StubFE* lhs, const StubFE* rhs) { return lhs->row() < rhs->row(); });
        std::deque<const StubFE*> toRemove;
        bool first(false);
        const StubFE* last = nullptr;
        for (const StubFE* stub : stubs) {
          if (!last) {
            last = stub;
            continue;
          }
          if (equal(stub, last)) {
            if (first)
              toRemove.push_back(stub);
            else
              first = true;
          } else
            first = false;
        }
        lost(toRemove, 0);
        // check for up to 4 stubs per 2bx packet
        if (static_cast<int>(stubs.size()) > setup_->mpaNumStub()) {
          lost(stubs, setup_->mpaNumStub());
          stubs.resize(setup_->mpaNumStub());
        }
        // store stubs for cic
        std::copy(stubs.begin(), stubs.end(), std::back_inserter(output));
      }
    }
  }

  // apply cbc rules, 3 stubs per bx row priority
  void DTC::cbc(const std::deque<const StubFE*>& input, std::deque<const StubFE*>& output, int iCIC) {
    auto begin = input.begin();
    for (int bx = 0; bx < setup_->cicNumBX(); bx++) {
      const auto end = std::find_if(begin, input.end(), [bx](const StubFE* s) { return s->bx() > bx; });
      // loop over 8 cbcs connected to one cic
      for (int iCBC = 0; iCBC < setup_->cicNumFEC(); iCBC++) {
        // collect 1bx packet stubs for this cbc
        std::deque<const StubFE*> stubs;
        std::copy_if(begin, end, std::back_inserter(stubs), [iCIC, iCBC](const StubFE* s) {
          return s->cic() == iCIC && s->fec() == iCBC;
        });
        begin = end;
        if (stubs.empty())
          continue;
        // check for only up to 1 stub per row, col combination, keep smallest abs(bend)
        auto equal = [](const StubFE* lhs, const StubFE* rhs) {
          return lhs->row() == rhs->row() && lhs->col() == rhs->col();
        };
        std::sort(stubs.begin(), stubs.end(), [](const StubFE* lhs, const StubFE* rhs) {
          return std::abs(lhs->bend()) < std::abs(rhs->bend());
        });
        std::stable_sort(
            stubs.begin(), stubs.end(), [](const StubFE* lhs, const StubFE* rhs) { return lhs->col() < rhs->col(); });
        std::stable_sort(
            stubs.begin(), stubs.end(), [](const StubFE* lhs, const StubFE* rhs) { return lhs->row() < rhs->row(); });
        const auto last = std::unique(stubs.begin(), stubs.end(), equal);
        lost(stubs, std::distance(stubs.begin(), last));
        stubs.erase(last, stubs.end());
        // check for up to 3 stubs per 1bx packet
        if (static_cast<int>(stubs.size()) > setup_->cbcNumStub()) {
          lost(stubs, setup_->cbcNumStub());
          stubs.resize(setup_->cbcNumStub());
        }
        // store stubs for cic
        std::copy(stubs.begin(), stubs.end(), std::back_inserter(output));
      }
    }
  }

  // emulate output
  void DTC::produce(const tt::Streams& input, tt::Streams& output) {
    // prep input data
    std::vector<std::deque<const StubFE*>> streamsFE(setup_->dtcNumModule());
    convert(input, streamsFE);
    // fill occopancy histos
    fill(streamsFE, 0);
    // convert front end stubs to global stubs
    std::vector<StubGL> stubsGL;
    auto acc = [](int sum, const std::vector<StubFE>& stubs) { return sum += stubs.size(); };
    const int size = std::accumulate(stubsFE_.begin(), stubsFE_.end(), 0, acc);
    stubsGL.reserve(size);
    std::vector<std::deque<const StubGL*>> streamsGL(setup_->dtcNumModule());
    produce(streamsFE, stubsGL, streamsGL);
    // emualte routing 9 links -> 8 single bx streams
    std::vector<std::deque<const StubGL*>> tmp8(setup_->tmp8NumChannel());
    unbox(streamsGL, tmp8);
    // fill occopancy histos
    fill(tmp8, 1);
    // emulate truncation
    truncate(tmp8, setup_->tmp8NumFrames());
    // emulate 8BX -> 12BX repacking
    std::vector<std::deque<const StubGL*>> tmp12(setup_->tmp12NumChannel());
    repack(tmp8, tmp12, setup_->tmp12NumNodes(), setup_->tmp12NumInputs(), setup_->tmp12NumOutputs());
    // fill occopancy histos
    fill(tmp12, 2);
    // emulate truncation
    truncate(tmp12, setup_->tmp12NumFrames());
    // emulate 12BX -> 18BX repacking
    std::vector<std::deque<const StubGL*>> tmp18(setup_->tmp18NumChannel());
    repack(tmp12, tmp18, setup_->tmp18NumNodes(), setup_->tmp18NumInputs(), setup_->tmp18NumOutputs());
    // fill occopancy histos
    fill(tmp18, 3);
    // emulate routing 2 -> 2 phi region splitting
    std::vector<StubDTC> stubsDTC;
    stubsDTC.reserve(setup_->sysNumOverlap() * size);
    std::vector<std::deque<const StubDTC*>> streamsDTC(setup_->tmp18NumChannel());
    produce(tmp18, stubsDTC, streamsDTC);
    // fill occopancy histos
    fill(streamsDTC, 4);
    // emulate truncation
    truncate(streamsDTC, setup_->sysNumFrames());
    // convert to streams
    convert(streamsDTC, output);
  }

  // read in input data
  void DTC::convert(const tt::Streams& input, std::vector<std::deque<const StubFE*>>& streamsFE) const {
    for (int channel = 0; channel < static_cast<int>(input.size()); channel++) {
      std::deque<const StubFE*>& stream = streamsFE[channel];
      std::deque<const StubFE*> stack;
      for (const tt::Frame& frame : input[channel]) {
        const TTBV ttBV(frame);
        TTBV lhs(ttBV, TTBV::S_, TTBV::S_ / 2);
        TTBV rhs(ttBV, TTBV::S_ / 2, 0);
        if (rhs.test(setup_->fePosValid())) {
          StubFE stubFE(setup_, channel, 1, rhs);
          const std::vector<StubFE>& stubs = stubsFE_[stubFE.bx()];
          stack.push_back(&*std::find(stubs.begin(), stubs.end(), stubFE));
        }
        if (lhs.test(setup_->fePosValid())) {
          StubFE stubFE(setup_, channel, 0, lhs);
          const std::vector<StubFE>& stubs = stubsFE_[stubFE.bx()];
          stream.push_back(&*std::find(stubs.begin(), stubs.end(), stubFE));
        } else
          stream.push_back(pop_front(stack));
      }
      stream.insert(stream.end(), stack.begin(), stack.end());
      // remove trailing gaps
      for (auto it = stream.end(); it != stream.begin();)
        it = (*--it) ? stream.begin() : stream.erase(it);
    }
  }

  // convert front end stubs to global stubs
  void DTC::produce(const std::vector<std::deque<const StubFE*>>& streamsFE,
                    std::vector<StubGL>& stubsGL,
                    std::vector<std::deque<const StubGL*>>& streamsGL) const {
    for (int iChannel = 0; iChannel < setup_->dtcNumModule(); iChannel++) {
      const std::deque<const StubFE*>& streamFE = streamsFE[iChannel];
      std::deque<const StubGL*>& streamGL = streamsGL[iChannel];
      for (const StubFE* stubFE : streamFE) {
        const StubGL* stubGL = nullptr;
        if (stubFE) {
          stubsGL.emplace_back(*stubFE);
          stubGL = &stubsGL.back();
        }
        streamGL.push_back(stubGL && stubGL->valid() ? stubGL : nullptr);
      }
    }
  }

  // convert stubs to streams
  void DTC::convert(const std::vector<std::deque<const StubDTC*>>& streamsDTC, tt::Streams& output) {
    for (int channel = 0; channel < static_cast<int>(output.size()); channel++) {
      tt::Stream& stream = output[channel];
      for (const StubDTC* stubDTC : streamsDTC[channel]) {
        if (!stubDTC) {
          stream.emplace_back(tt::Frame());
          continue;
        }
        stream.push_back(stubDTC->frame().second);
      }
    }
  }

  // emualte routing 9 links -> 8 single bx streams
  void DTC::unbox(const std::vector<std::deque<const StubGL*>>& in, std::vector<std::deque<const StubGL*>>& out) const {
    for (int node = 0; node < setup_->tmp8NumNodes(); node++) {
      const int offsetIn = node * setup_->tmp8NumInputs();
      const int offsetOut = node * setup_->tmp8NumOutputs();
      for (int bx = 0; bx < setup_->tmp8NumOutputs(); bx++) {
        std::vector<std::deque<const StubGL*>> inputs(setup_->tmp8NumInputs());
        for (int chan = 0; chan < setup_->tmp8NumInputs(); chan++) {
          std::deque<const StubGL*>& input = inputs[chan];
          for (const StubGL* stub : in[offsetIn + chan])
            input.push_back(stub && stub->bx() == bx ? stub : nullptr);
        }
        merge(inputs, out[offsetOut + bx], true);
      }
    }
  }

  // emulate 8BX -> 12BX repacking and 12BX -> 18BX
  void DTC::repack(const std::vector<std::deque<const StubGL*>>& in,
                   std::vector<std::deque<const StubGL*>>& out,
                   int numNodes,
                   int numInputs,
                   int numOutputs) const {
    for (int node = 0; node < numNodes; node++) {
      const int offsetNodeIn = node * numInputs;
      const int offsetNodeOut = node * numOutputs;
      for (int chan = 0; chan < numOutputs; chan++) {
        const int offset = offsetNodeIn + chan;
        std::vector<std::deque<const StubGL*>> inputs({in[offset], in[offset + numOutputs]});
        merge(inputs, out[offsetNodeOut + chan]);
      }
    }
  }

  // emulate routing 2 -> 2 phi region splitting
  void DTC::produce(const std::vector<std::deque<const StubGL*>>& in,
                    std::vector<StubDTC>& stubs,
                    std::vector<std::deque<const StubDTC*>>& out) const {
    for (int node = 0; node < setup_->tmp18NumOutputs(); node++) {
      const int offset = node * setup_->sysNumOverlap();
      for (int overlap = 0; overlap < setup_->sysNumOverlap(); overlap++) {
        std::vector<std::deque<const StubDTC*>> inputs(setup_->sysNumOverlap());
        for (int chan = 0; chan < setup_->sysNumOverlap(); chan++) {
          std::deque<const StubDTC*>& stream = inputs[chan];
          for (const StubGL* stub : in[node + chan * setup_->tmp18NumOutputs()]) {
            if (!stub || !stub->overlap().test(overlap)) {
              stream.push_back(nullptr);
              continue;
            }
            stubs.emplace_back(*stub, overlap);
            const StubDTC* stubDTC = &stubs.back();
            stream.push_back(stubDTC->valid() ? stubDTC : nullptr);
          }
        }
        merge(inputs, out[offset + overlap]);
      }
    }
  }

  // clock accurate firmware emulation, each while trip describes one clock tick, one stub in and one stub out per tick
  template <class Stub>
  void DTC::merge(std::vector<std::deque<const Stub*>>& in, std::deque<const Stub*>& out, bool en) const {
    auto empty = [](const std::deque<const Stub*>& stream) { return stream.empty(); };
    std::vector<std::deque<const Stub*>> stacks(in.size());
    while (!std::all_of(in.begin(), in.end(), empty) || !std::all_of(stacks.begin(), stacks.end(), empty)) {
      // fill input fifos
      int index(0);
      for (std::deque<const Stub*>& stack : stacks) {
        const Stub* stub = pop_front(in[index++]);
        if (!stub)
          continue;
        // buffer overflow
        if (en && static_cast<int>(stack.size()) == setup_->unDepth() - 1)
          pop_front(stack);
        stack.push_back(stub);
      }
      // merge input fifos to one stream, prioritizing lower input channel over higher channel
      bool nothingToRoute(true);
      for (std::deque<const Stub*>& stack : stacks) {
        const Stub* stub = pop_front(stack);
        if (!stub)
          continue;
        nothingToRoute = false;
        out.push_back(stub);
        break;
      }
      if (nothingToRoute)
        out.push_back(nullptr);
    }
  }

  // fill occopancy histos
  template <class Stub>
  void DTC::fill(std::vector<std::deque<const Stub*>>& streams, int step) {
    for (int channel = 0; channel < static_cast<int>(streams.size()); channel++) {
      const int size = streams[channel].size();
      his_[step]->Fill(size);
      prof_[step]->Fill(channel, size);
    }
  }

  // fill lost stubs histo
  void DTC::lost(const std::deque<const StubFE*>& stubs, int limit) {
    for (auto it = std::next(stubs.begin(), limit); it != stubs.end(); it++) {
      const StubFE* stub = *it;
      if (!stub)
        continue;
      const StubGL stubGL(*stub);
      hisRZLost_->Fill(stubGL.z(), stubGL.r() + setup_->regChosenRofPhi());
    }
  }
  // fill lost stubs histo
  void DTC::lost(const std::deque<const StubGL*>& stubs, int limit) {
    for (auto it = std::next(stubs.begin(), limit); it != stubs.end(); it++) {
      const StubGL* stub = *it;
      if (!stub)
        continue;
      hisRZLost_->Fill(stub->z(), stub->r() + setup_->regChosenRofPhi());
    }
  }
  // fill lost stubs histo
  void DTC::lost(const std::deque<const StubDTC*>& stubs, int limit) {
    for (auto it = std::next(stubs.begin(), limit); it != stubs.end(); it++) {
      const StubDTC* stub = *it;
      if (!stub)
        continue;
      hisRZLost_->Fill(stub->stubGL()->z(), stub->stubGL()->r() + setup_->regChosenRofPhi());
    }
  }

  // emulate truncation
  template <class Stub>
  void DTC::truncate(std::vector<std::deque<const Stub*>>& streams, int size) {
    for (std::deque<const Stub*>& stream : streams) {
      if (static_cast<int>(stream.size()) > size) {
        lost(stream, size);
        stream.resize(size);
      }
      // remove trailing gaps
      for (auto it = stream.end(); it != stream.begin();)
        it = (*--it) ? stream.begin() : stream.erase(it);
    }
  }

  // pop_front function which additionally returns copy of deleted front
  template <class Stub>
  const Stub* DTC::pop_front(std::deque<const Stub*>& stream) const {
    const Stub* stub = nullptr;
    if (!stream.empty()) {
      stub = stream.front();
      stream.pop_front();
    }
    return stub;
  }

}  // namespace trackerDTC
