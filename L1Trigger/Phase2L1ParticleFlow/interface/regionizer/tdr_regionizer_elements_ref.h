#ifndef tdr_regionizer_elements_ref_h
#define tdr_regionizer_elements_ref_h

#include "DataFormats/L1TParticleFlow/interface/layer1_emulator.h"

#include <vector>
#include <map>
#include <cassert>
#include <algorithm>

#include "L1Trigger/Phase2L1ParticleFlow/interface/dbgPrintf.h"

namespace l1ct {
  namespace tdr_regionizer {

    inline int dphi_wrap(int local_phi) {
      if (local_phi > l1ct::Scales::INTPHI_PI)
        local_phi -= l1ct::Scales::INTPHI_TWOPI;
      else if (local_phi <= -l1ct::Scales::INTPHI_PI)
        local_phi += l1ct::Scales::INTPHI_TWOPI;
      return local_phi;
    }

    // These are done per-sector (= per-input link)
    template <typename T>
    class PipeObject {
    public:
      PipeObject() {}
      PipeObject(const T& obj, std::vector<size_t> srIndices, int glbphi, int glbeta, unsigned int clk);

      unsigned int getClock() const { return linkobjclk_; }
      void setClock(unsigned int clock) { linkobjclk_ = clock; }
      const std::vector<size_t>& getSRIndices() const { return srIndices_; }
      size_t getNextSRIndex() const { return srIndices_.at(objcount_); }
      unsigned int getCount() const { return objcount_; }
      void incCount() { objcount_++; }
      int getPt() const { return obj_.hwPt.to_int(); }
      int getGlbPhi() const { return glbphi_; }
      int getGlbEta() const { return glbeta_; }

      T& getRawObj() { return obj_; }
      const T& getRawObj() const { return obj_; }

    private:
      T obj_;
      /// the SR linearized indices (can index regionmap_) where this object needs to go
      std::vector<size_t> srIndices_;
      /// The global eta and phi of the object (somewhat redundant with obj_)
      int glbeta_, glbphi_;
      unsigned int linkobjclk_, objcount_;
    };

    template <typename T>
    class Pipe {
    public:
      Pipe(unsigned int nphi = 9) : clkindex_(0), nphi_(nphi) {}

      void addObj(T obj, std::vector<size_t> srs, int glbeta, int glbphi);

      PipeObject<T>& getObj(unsigned int index = 0) { return data_[index]; }
      const PipeObject<T>& getObj(unsigned int index = 0) const { return data_[index]; }

      unsigned int getClock(unsigned int index = 0) const { return getObj(index).getClock(); }
      void setClock(unsigned int clock, unsigned int index = 0) { return getObj(index).setClock(clock); }
      unsigned int getCount(unsigned int index = 0) const { return getObj(index).getCount(); }
      void incCount(unsigned int index = 0) { getObj(index).incCount(); }
      void erase(unsigned int index = 0) { data_.erase(data_.begin() + index); }
      int getPt(unsigned int index = 0) const { return getObj(index).getPt(); }
      int getGlbPhi(unsigned int index = 0) const { return getObj(index).getGlbPhi(); }
      int getGlbEta(unsigned int index = 0) const { return getObj(index).getGlbEta(); }

      int getClosedIndexForObject(unsigned int index = 0);
      /// This returns the hardware pipe index (since there are one per SR pair)
      size_t getPipeIndexForObject(unsigned int index = 0);

      unsigned int getPipeSize() const { return data_.size(); }

      void reset() {
        clkindex_ = 0;
        data_.clear();
      }

    private:
      unsigned int clkindex_, nphi_;
      std::vector<PipeObject<T>> data_;
    };

    template <typename T>
    class Regionizer {
    public:
      Regionizer() {}
      Regionizer(unsigned int neta,
                 unsigned int nphi,      //the number of eta and phi SRs in a big region (board)
                 unsigned int nregions,  // The total number of small regions in the full barrel
                 unsigned int maxobjects,
                 int bigRegionMin,
                 int bigRegionMax,  // the phi range covered by this board
                 int nclocks);

      void initSectors(const std::vector<DetectorSector<T>>& sectors);
      void initSectors(const DetectorSector<T>& sector);
      void initRegions(const std::vector<PFInputRegion>& regions);

      // is the given small region in the big region
      bool isInBigRegion(const PFRegionEmu& reg) const;

      unsigned int getSize() const { return pipes_.size(); }
      unsigned int getPipeSize(unsigned int linkIndex) const { return pipes_[linkIndex].getPipeSize(); }

      std::vector<size_t> getSmallRegions(int glbeta, int glbphi) const;

      void addToPipe(const T& obj, unsigned int index);
      void setPipe(const std::vector<T>& objvec, unsigned int index);
      void setPipes(const std::vector<std::vector<T>>& objvecvec);

      // linkIndex == sector
      int getPipeTime(int linkIndex, int linkTimeOfObject, int linkAlgoClockRunningTime);

      /// This either removes the next object on the link or inrements the count; It returns the next time
      int popLinkObject(int linkIndex, int currentTimeOfObject);
      int timeNextFromIndex(unsigned int linkIndex, int time) {
        return getPipeTime(linkIndex, pipes_[linkIndex].getClock(), time);
      }

      void initTimes();

      int getClosedIndexForObject(unsigned int linknum, unsigned int index = 0) {
        return pipes_[linknum].getClosedIndexForObject(index);
      }

      /// This retruns the linearized small region associated with the given item
      size_t getPipeIndexForObject(unsigned int linknum, unsigned int index = 0) {
        return pipes_[linknum].getPipeIndexForObject(index);
      }

      /// This returns the hardware pipe number of the item. Generally two SRs share a pipe
      size_t getHardwarePipeIndexForObject(unsigned int linknum, unsigned int index = 0) {
        return getHardwarePipeIndex(getPipeIndexForObject(linknum, index));
      }

      /// 'put' object in small region
      void addToSmallRegion(unsigned int linkNum, unsigned int index = 0);

      void run(bool debug = false);

      void reset();

      /// Return a map of of the SRs indexed by SR index (covering only those from board)
      std::map<size_t, std::vector<T>> fillRegions(bool doSort);

      void printDebug(int count) const;

    private:
      /// SRs share RAMs (and hardware pipes)
      static size_t constexpr SRS_PER_RAM = 2;

      /// Because some SRs share pipes, this determines the pipe index for a linearize SR index
      /// (This is based on the VHDL function, get_target_pipe_index_subindex)
      size_t getHardwarePipeIndex(size_t srIndex) const { return srIndex / SRS_PER_RAM; }

      // this function is for sorting small regions first in phi and then in eta.
      // It takes regions_ indices
      bool sortRegionsRegular(size_t a, size_t b) const;

      /// The numbers of eta and phi in a big region (board)
      unsigned int neta_, nphi_;
      /// The total number of small regions in the barrel (not just in the board)
      unsigned int nregions_;
      /// The maximum number of objects to output per small region
      unsigned int maxobjects_;
      /// The number of input sectors for this type of device
      unsigned int nsectors_;
      /// the minimumum phi of this board
      int bigRegionMin_;
      /// the maximum phi of this board
      int bigRegionMax_;
      /// the number of clocks to receive one event
      int nclocks_;

      /// the region information assopciated with each input sector
      std::vector<l1ct::PFRegionEmu> sectors_;

      /// the region information associated with each SR
      std::vector<l1ct::PFRegionEmu> regions_;

      /// indices of regions that are in the big region (board)
      std::vector<size_t> regionmap_;

      /// One pipe per each sector (link). These do not correspond to the firmware pipes
      std::vector<Pipe<T>> pipes_;
      /// One entry per sector (= link = pipe). If the pipe is empty, this is always -1
      std::vector<int> timeOfNextObject_;

      /// The objects in each small region handled in board; Indexing corresponds to that in regionmap_
      std::vector<std::vector<T>> smallRegionObjects_;
    };

  }  // namespace  tdr_regionizer
}  // namespace l1ct

#endif
