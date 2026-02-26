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

    /// the components that make up the L1 regionizer buffer
    template <typename T>
    class BufferEntry {
    public:
      BufferEntry() {}
      BufferEntry(const T& obj, int glbeta, int glbphi);

      int pt() const { return obj_.intPt(); }
      int glbPhi() const { return glbphi_; }
      int glbEta() const { return glbeta_; }

      //T obj() { return obj_; }
      const T& obj() const { return obj_; }

    private:
      T obj_;
      /// the SR linearized indices (can index regionmap_) where this object needs to go
      int glbeta_, glbphi_;
    };

    /// The L1 regionizer buffer (corresponding to level1_fifo_buffer.vhd)
    template <typename T>
    class Buffer {
    public:
      Buffer() {}

      void addEntry(const T& obj,
                    std::vector<size_t> srs,  // LOGICAL SRs
                    int glbeta,
                    int glbphi);

      bool empty(size_t sr) const;
      BufferEntry<T> getEntry(size_t sr);

      // mainly for debug/validation, to check that nothing is left over after an event
      // This is the number of entries in all the deques, not necessarily the number of
      // entries in the map.
      unsigned int numEntries() const;

      void reset() { data_.clear(); }

      void printDebug(size_t bufIdx, size_t logBufIdx) const;

    private:
      /// The actual data, indexed by the physical SR
      std::map<size_t, std::deque<BufferEntry<T>>> data_;
    };

    template <typename T>
    class Regionizer {
    public:
      Regionizer() = delete;
      Regionizer(unsigned int maxobjects, bool debug = false);

      void initSectors(const std::vector<DetectorSector<T>>& sectors);
      void initSectors(const DetectorSector<T>& sector);
      void initRegions(const std::vector<PFInputRegion>& regions);

      void fillBuffers(const std::vector<DetectorSector<T>>& sectors);
      void fillBuffers(const DetectorSector<T>& sector);

      void run();

      void reset();

      const std::vector<std::vector<T>>& smallRegions() const { return smallRegionObjects_; }
      void clearSmallRegions();

      void printDebug(int count) const;

    private:
      const size_t SMALL_REGION_ETA_COUNT = 6;
      const size_t SMALL_REGION_PHI_COUNT = 9;

      /// @brief convert to logical SR
      /// @param sr linear small region
      /// @return pair (eta, phi) region index)
      std::pair<size_t, size_t> get_small_region(size_t sr) const;

      /// logical links associated with a logical sr (customized for each type)
      std::vector<size_t> linksForSR(size_t sr) const;
      std::vector<size_t> caloLinksHelper(size_t iphi) const;

      unsigned int numBuffers() const { return buffers_.size(); }
      unsigned int numEntries(unsigned int bufferIndex) const { return buffers_[bufferIndex].numEntries(); }

      std::vector<size_t> getSmallRegions(int glbeta, int glbphi) const;

      /// fill the buffers with an event's link data. The buffers should be empty--checked by assert
      void setBuffers(const std::vector<std::vector<T>>&& objvecvec);

      /// 'put' object in small region. The sr is physics
      void addToSmallRegion(size_t sr, BufferEntry<T>&& bufEntry);

      /// returns 2D arrays, sectors (links) first dimension, objects second
      std::vector<std::vector<T>> fillLinks(const std::vector<DetectorSector<T>>& sectors) const;
      std::vector<std::vector<T>> fillLinks(const DetectorSector<T>& sector) const;

      // this function is for sorting small regions first in phi and then in eta.
      // It takes regions_ indices
      bool sortRegionsRegular(size_t a, size_t b) const;
      bool sortRegionsHelper(int etaa, int etab, int phia, int phib) const;

      // This is for sorting sectors. It sorts in eta first, then phi
      bool sortSectors(size_t a, size_t b) const;
      bool sortSectorsHelper(int etaa, int etab, int phia, int phib) const;

      /// get the index in regions_ for a particular SR.
      size_t regionIndex(int sr) const { return regionmap_.at(sr); }

      /// get the logical buffer index (i.e. the index in the order in the firmware)
      size_t logicBuffIndex(size_t bufIdx) const;

      /// The maximum number of objects to output per small region
      unsigned int maxobjects_;

      /// the region information associated with each input sector (link). Note, this is indexed in physical order
      std::vector<l1ct::PFRegionEmu> sectors_;

      /// the region information associated with each SR
      std::vector<l1ct::PFRegionEmu> regions_;

      /// indices of regions that are in the big region (board)
      std::vector<size_t> regionmap_;

      /// indices maps the sectors from the way they appear in the software to the (logical) order they are done in the regionizer firmware
      std::vector<size_t> sectorMapPhysToLog_;

      /// the inverse mapping of sectormap_ (only used for debug printing)
      std::vector<size_t> sectorMapLogToPhys_;

      /// The buffers. There is one buffer per link (or sector). Note, this is also indexed in physical order (= same index as sectors)
      std::vector<Buffer<T>> buffers_;

      /// The objects in each small region handled in board; Indexing corresponds to that in regionmap_
      std::vector<std::vector<T>> smallRegionObjects_;

      bool debug_;
    };

  }  // namespace  tdr_regionizer
}  // namespace l1ct

#endif
