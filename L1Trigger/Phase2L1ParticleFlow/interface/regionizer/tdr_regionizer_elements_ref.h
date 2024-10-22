#ifndef tdr_regionizer_elements_ref_h
#define tdr_regionizer_elements_ref_h

#include "DataFormats/L1TParticleFlow/interface/layer1_emulator.h"

#include <vector>
#include <map>
#include <deque>
#include <cassert>
#include <algorithm>

#include "L1Trigger/Phase2L1ParticleFlow/interface/dbgPrintf.h"

namespace l1ct {
  namespace tdr_regionizer {

    inline int phi_wrap(int local_phi) {
      if (local_phi > l1ct::Scales::INTPHI_PI)
        local_phi -= l1ct::Scales::INTPHI_TWOPI;
      else if (local_phi <= -l1ct::Scales::INTPHI_PI)
        local_phi += l1ct::Scales::INTPHI_TWOPI;
      return local_phi;
    }

    /// corresponds to level1_to_2_pipe_t in firmware
    template <typename T>
    class PipeEntry {
    public:
      PipeEntry() : obj_(), sr_(-1) {}
      PipeEntry(const T& obj, int sr, int glbeta, int glbphi) : obj_(obj), sr_(sr), glbeta_(glbeta), glbphi_(glbphi) {}

      int sr() const { return sr_; }

      // Note, this returns a copy so you can modify
      T obj() const { return obj_; }

      bool valid() const { return sr_ >= 0; }

      void setInvalid() { sr_ = -1; }

      int pt() const { return obj_.intPt(); }
      int glbPhi() const { return glbphi_; }
      int glbEta() const { return glbeta_; }

    private:
      T obj_;
      /// the SR linearized indices (can index regionmap_) where this object needs to go; -1 means invalid
      int sr_;
      /// The global eta and phi of the object (hard to get with duplicates)
      int glbeta_, glbphi_;
    };

    /// The pipe, with multiple inputs and one output
    template <typename T>
    class Pipe {
    public:
      /// if using the default constructor, have to call setTaps before use
      Pipe() : pipe_() {}

      Pipe(size_t ntaps) : pipe_(ntaps) {}

      void setTaps(size_t taps) { pipe_.resize(taps); }

      /// check if the entry is valid (i.e. already has data)
      bool valid(size_t idx) const { return pipe_.at(idx).valid(); }

      /// should check if valid before adding an entry
      void addEntry(size_t idx, const PipeEntry<T>& entry) { pipe_[idx] = entry; }

      /// perform one tick, shifting all the entries to higher indices, and returning the last
      PipeEntry<T> popEntry();

      void reset();

      size_t size() const { return pipe_.size(); }

      /// for debug
      const PipeEntry<T>& entry(size_t idx) const { return pipe_[idx]; }

    private:
      std::vector<PipeEntry<T>> pipe_;
    };

    /// The pipe, with multiple inputs and one output
    template <typename T>
    class Pipes {
    public:
      /// the number of pipes
      Pipes(size_t nregions) : pipes_(nregions / SRS_PER_RAM) {}

      /// set the number of taps in each pipe
      void setTaps(size_t taps);

      /// check if the entry is valid (i.e. already has data)
      bool valid(int sr, size_t logicBufIdx) const { return pipes_[pipeIndex(sr)].valid(logicBufIdx); }

      /// should check if valid before adding an entry
      void addEntry(int sr, size_t logicBufIdx, const PipeEntry<T>& entry) {
        pipes_[pipeIndex(sr)].addEntry(logicBufIdx, entry);
      }

      /// perform one tick, shifting all the entries to higher indices, and returning the last
      PipeEntry<T> popEntry(size_t pipe) { return pipes_[pipe].popEntry(); };

      void reset();

      size_t size() const { return pipes_.size(); }

      size_t numTaps() const { return pipes_.at(0).size(); }

      /// for debug
      const PipeEntry<T>& entry(size_t pipe, size_t tap) const { return pipes_[pipe].entry(tap); }

    private:
      /// SRs share RAMs (and hardware pipes)
      static size_t constexpr SRS_PER_RAM = 2;

      /// Because some SRs share pipes, this determines the pipe index for a linearize SR index
      /// (This is based on the VHDL function, get_target_pipe_index_subindex)
      size_t pipeIndex(int sr) const { return sr / SRS_PER_RAM; }

      std::vector<Pipe<T>> pipes_;
    };

    /// the components that make up the L1 regionizer buffer
    template <typename T>
    class BufferEntry {
    public:
      BufferEntry() {}
      BufferEntry(const T& obj, std::vector<size_t> srIndices, int glbeta, int glbphi, unsigned int clk);

      unsigned int clock() const { return linkobjclk_; }
      int nextSR() const { return (objcount_ < srIndices_.size()) ? srIndices_[objcount_] : -1; }
      void incSR() { objcount_++; }
      int pt() const { return obj_.intPt(); }
      int glbPhi() const { return glbphi_; }
      int glbEta() const { return glbeta_; }

      //T obj() { return obj_; }
      const T& obj() const { return obj_; }

    private:
      T obj_;
      /// the SR linearized indices (can index regionmap_) where this object needs to go
      std::vector<size_t> srIndices_;
      /// The global eta and phi of the object (hard to get with duplicates)
      int glbeta_, glbphi_;
      unsigned int linkobjclk_, objcount_;
    };

    /// The L1 regionizer buffer (corresponding to level1_fifo_buffer.vhd)
    template <typename T>
    class Buffer {
    public:
      Buffer() : clkindex360_(INIT360), clkindex240_(INIT240), timeOfNextObject_(-1) {}

      void addEntry(
          const T& obj, std::vector<size_t> srs, int glbeta, int glbphi, unsigned int dupNum, unsigned int ndup);

      BufferEntry<T>& front() { return data_.front(); }
      const BufferEntry<T>& front() const { return data_.front(); }

      /// sets the next time something is taken from this buffer
      void updateNextObjectTime(int currentTime);

      /// delete the front element
      void pop() { data_.pop_front(); }

      // mostly for debug
      unsigned int clock(unsigned int index = 0) const { return data_[index].clock(); }
      int pt(unsigned int index = 0) const { return data_[index].pt(); }
      int glbPhi(unsigned int index = 0) const { return data_[index].glbPhi(); }
      int glbEta(unsigned int index = 0) const { return data_[index].glbEta(); }

      unsigned int numEntries() const { return data_.size(); }

      /// pop the first entry, formatted for inclusion in pipe
      PipeEntry<T> popEntry(int currTime, bool debug);

      int timeOfNextObject() const { return timeOfNextObject_; }

      void reset() {
        clkindex360_ = INIT360;
        clkindex240_ = INIT240;
        data_.clear();
        timeOfNextObject_ = -1;
      }

    private:
      // used when building up the linkobjclk_ entries for the BufferEntries
      unsigned int nextObjClk(unsigned int ndup);

      // transient--used only during event construction, not used after
      // Counts in 1.39ns increments (i.e. 360 increments by 2, 240 by 3)
      unsigned int clkindex360_;
      unsigned int clkindex240_;

      static unsigned int constexpr INIT360 = 1;
      static unsigned int constexpr INIT240 = 0;

      /// The actual data
      std::deque<BufferEntry<T>> data_;

      /// the time of the next object in the buffer (-1 if none)
      int timeOfNextObject_;
    };

    template <typename T>
    class Regionizer {
    public:
      Regionizer() = delete;
      Regionizer(unsigned int neta,
                 unsigned int nphi,  //the number of eta and phi SRs in a big region (board)
                 unsigned int maxobjects,
                 int bigRegionMin,
                 int bigRegionMax,  // the phi range covered by this board
                 unsigned int nclocks,
                 unsigned int ndup = 1,  // how much one duplicates the inputs (to increase processing bandwidth)
                 bool debug = false);

      void initSectors(const std::vector<DetectorSector<T>>& sectors);
      void initSectors(const DetectorSector<T>& sector);
      void initRegions(const std::vector<PFInputRegion>& regions);

      void fillBuffers(const std::vector<DetectorSector<T>>& sectors);
      void fillBuffers(const DetectorSector<T>& sector);

      void run();

      void reset();

      /// Return a map of of the SRs indexed by SR index (covering only those from board)
      std::map<size_t, std::vector<T>> fillRegions(bool doSort);

      void printDebug(int count) const;

    private:
      /// is the given small region in the big region
      bool isInBigRegion(const PFRegionEmu& reg) const;

      /// Does the given region fit in the big region, taking into account overlaps?
      bool isInBigRegionLoose(const PFRegionEmu& reg) const;

      unsigned int numBuffers() const { return buffers_.size(); }
      unsigned int numEntries(unsigned int bufferIndex) const { return buffers_[bufferIndex].numEntries(); }

      std::vector<size_t> getSmallRegions(int glbeta, int glbphi) const;

      void addToBuffer(const T& obj, unsigned int index, unsigned int dupNum);
      void setBuffer(const std::vector<T>& objvec, unsigned int index);
      void setBuffers(const std::vector<std::vector<T>>&& objvecvec);

      /// This retruns the linearized small region associated with the given item (-1 is throwout)
      int nextSR(unsigned int linknum, unsigned int index = 0) { return buffers_[linknum].nextSR(index); }

      /// 'put' object in small region
      void addToSmallRegion(PipeEntry<T>&&);

      /// returns 2D arrays, sectors (links) first dimension, objects second
      std::vector<std::vector<T>> fillLinks(const std::vector<DetectorSector<T>>& sectors) const;
      std::vector<std::vector<T>> fillLinks(const DetectorSector<T>& sector) const;

      // this function is for sorting small regions first in phi and then in eta.
      // It takes regions_ indices
      bool sortRegionsRegular(size_t a, size_t b) const;

      bool sortSectors(size_t a, size_t b) const;

      bool sortRegionsHelper(int etaa, int etab, int phia, int phib) const;

      /// get the index in regions_ for a particular SR.
      size_t regionIndex(int sr) const { return regionmap_.at(sr); }

      /// get the logical buffer index (i.e. the index in the order in the firmware)
      size_t logicBuffIndex(size_t bufIdx) const;

      /// The numbers of eta and phi in a big region (board)
      unsigned int neta_, nphi_;
      /// The maximum number of objects to output per small region
      unsigned int maxobjects_;
      /// The number of input sectors for this type of device
      unsigned int nsectors_;
      /// the minimumum phi of this board
      int bigRegionMin_;
      /// the maximum phi of this board
      int bigRegionMax_;
      /// the number of clocks to receive one event
      unsigned int nclocks_;
      /// How many buffers per link (default 1)
      unsigned int ndup_;

      /// the region information assopciated with each input sector
      std::vector<l1ct::PFRegionEmu> sectors_;

      /// the region information associated with each SR
      std::vector<l1ct::PFRegionEmu> regions_;

      /// indices of regions that are in the big region (board)
      std::vector<size_t> regionmap_;

      /// indices maps the sectors from the way they appear in the software to the order they are done in the regionizer
      std::vector<size_t> sectormap_;

      /// the inverse mapping of sectormap_ (only used for debug printing)
      std::vector<size_t> invsectormap_;

      /// The buffers. There are ndup_ buffers per link/sector
      std::vector<Buffer<T>> buffers_;

      /// The pipes, one per ram (see SRS_PER_RAM)
      Pipes<T> pipes_;

      /// The objects in each small region handled in board; Indexing corresponds to that in regionmap_
      std::vector<std::vector<T>> smallRegionObjects_;

      /// Whether this is the first event (since timing is a bit different then)
      bool firstEvent_;

      /// This is the delay (only applied after first event) before processing starts
      static unsigned int constexpr DELAY_TO_START = 10;

      bool debug_;
    };

  }  // namespace  tdr_regionizer
}  // namespace l1ct

#endif
