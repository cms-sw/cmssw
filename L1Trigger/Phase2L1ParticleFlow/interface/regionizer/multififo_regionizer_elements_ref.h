#ifndef multififo_regionizer_elements_ref_h
#define multififo_regionizer_elements_ref_h

#include "DataFormats/L1TParticleFlow/interface/layer1_emulator.h"

#include <list>
#include <vector>
#include <deque>
#include <cassert>

namespace l1ct {
  namespace multififo_regionizer {
    template <typename T>
    inline void shift(T& from, T& to) {
      to = from;
      from.clear();
    }
    template <typename TL, typename T>
    inline void pop_back(TL& from, T& to) {
      assert(!from.empty());
      to = from.back();
      from.pop_back();
    }

    inline int dphi_wrap(int local_phi) {
      if (local_phi > l1ct::Scales::INTPHI_PI)
        local_phi -= l1ct::Scales::INTPHI_TWOPI;
      else if (local_phi <= -l1ct::Scales::INTPHI_PI)
        local_phi += l1ct::Scales::INTPHI_TWOPI;
      return local_phi;
    }

    template <typename T>
    inline void push_to_fifo(const T& t, int local_eta, int local_phi, std::list<T>& fifo) {
      fifo.push_front(t);
      fifo.front().hwEta = local_eta;
      fifo.front().hwPhi = local_phi;
    }

    template <typename T>
    inline void maybe_push(const T& t,
                           const l1ct::PFRegionEmu& sector,
                           const l1ct::PFRegionEmu& region,
                           std::list<T>& fifo,
                           bool useAlsoVtxCoords);
    template <>
    inline void maybe_push<l1ct::TkObjEmu>(const l1ct::TkObjEmu& t,
                                           const l1ct::PFRegionEmu& sector,
                                           const l1ct::PFRegionEmu& region,
                                           std::list<l1ct::TkObjEmu>& fifo,
                                           bool useAlsoVtxCoords);

    template <typename T>
    class RegionBuffer {
    public:
      RegionBuffer() : nfifos_(0) {}
      void initFifos(unsigned int nfifos);
      void initRegion(const l1ct::PFRegionEmu& region, bool useAlsoVtxCoords) {
        region_ = region;
        useAlsoVtxCoords_ = useAlsoVtxCoords;
      }
      void flush();
      void maybe_push(int fifo, const T& t, const l1ct::PFRegionEmu& sector);
      T pop();

    private:
      unsigned int nfifos_;
      bool useAlsoVtxCoords_;
      l1ct::PFRegionEmu region_;
      std::vector<std::list<T>> fifos_;
      std::vector<std::pair<std::vector<T>, std::vector<T>>> queues_;

      T pop_next_trivial_();
      void fifos_to_stage_(std::vector<T>& staging_area);
      void queue_to_stage_(std::vector<T>& queue, std::vector<T>& staging_area);
      void stage_to_queue_(std::vector<T>& staging_area, std::vector<T>& queue);
      T pop_queue_(std::vector<T>& queue);
    };

    template <typename T>
    inline bool local_eta_phi_window(const T& t,
                                     const l1ct::glbeta_t& etaMin,
                                     const l1ct::glbeta_t& etaMax,
                                     const l1ct::glbphi_t& phiMin,
                                     const l1ct::glbphi_t& phiMax);
    template <>
    inline bool local_eta_phi_window<l1ct::TkObjEmu>(const l1ct::TkObjEmu& t,
                                                     const l1ct::glbeta_t& etaMin,
                                                     const l1ct::glbeta_t& etaMax,
                                                     const l1ct::glbphi_t& phiMin,
                                                     const l1ct::glbphi_t& phiMax);

    template <typename T>
    class EtaPhiBuffer {
    public:
      EtaPhiBuffer() {}
      EtaPhiBuffer(unsigned int maxitems,
                   const l1ct::glbeta_t& etaMin = 0,
                   const l1ct::glbeta_t& etaMax = 0,
                   const l1ct::glbeta_t& etaShift = 0,
                   const l1ct::glbphi_t& phiMin = 0,
                   const l1ct::glbphi_t& phiMax = 0,
                   const l1ct::glbphi_t& phiShift = 0)
          : size_(maxitems),
            iwrite_(0),
            iread_(0),
            etaMin_(etaMin),
            etaMax_(etaMax),
            etaShift_(etaShift),
            phiMin_(phiMin),
            phiMax_(phiMax),
            phiShift_(phiShift) {}
      void maybe_push(const T& t);
      void writeNewEvent() {
        iwrite_ = 1 - iwrite_;
        items_[iwrite_].clear();
      }
      void readNewEvent() { iread_ = 1 - iread_; }
      T pop();
      unsigned int writeSize() const { return items_[iwrite_].size(); }
      unsigned int readSize() const { return items_[iread_].size(); }
      unsigned int maxSize() const { return size_; }
      void reset();

    private:
      unsigned int size_, iwrite_, iread_;
      l1ct::glbeta_t etaMin_, etaMax_;
      l1ct::glbeta_t etaShift_;
      l1ct::glbphi_t phiMin_, phiMax_;
      l1ct::glbphi_t phiShift_;
      std::deque<T> items_[2];
    };

    // forward decl for later
    template <typename T>
    class RegionMux;

    template <typename T>
    class RegionBuilder {
    public:
      RegionBuilder() {}
      RegionBuilder(unsigned int iregion, unsigned int nsort) : iregion_(iregion), sortbuffer_(nsort) {}
      void push(const T& in);
      void pop(RegionMux<T>& out);

    private:
      unsigned int iregion_;
      std::vector<T> sortbuffer_;
    };

    template <typename T>
    class RegionMux {
    public:
      RegionMux() : nregions_(0) {}
      RegionMux(unsigned int nregions,
                unsigned int nsort,
                unsigned int nout,
                bool streaming,
                unsigned int outii = 0,
                unsigned int pauseii = 0)
          : nregions_(nregions),
            nsort_(nsort),
            nout_(nout),
            outii_(outii),
            pauseii_(pauseii),
            streaming_(streaming),
            buffer_(nregions * nsort),
            iter_(0),
            ireg_(nregions) {
        assert(streaming ? (outii * nout >= nsort) : (nout == nsort));
        for (auto& t : buffer_)
          t.clear();
      }
      void push(unsigned int region, std::vector<T>& in);
      bool stream(bool newevt, std::vector<T>& out);

    private:
      unsigned int nregions_, nsort_, nout_, outii_, pauseii_;
      bool streaming_;
      std::vector<T> buffer_;
      unsigned int iter_, ireg_;
    };

    // out of the Regionizer<T> since it doesn't depend on T and may be shared
    struct Route {
      unsigned short int sector, link, region, fifo;
      Route(unsigned short int from_sector,
            unsigned short int from_link,
            unsigned short int to_region,
            unsigned short int to_fifo)
          : sector(from_sector), link(from_link), region(to_region), fifo(to_fifo) {}
    };

    template <typename T>
    class Regionizer {
    public:
      Regionizer() {}
      Regionizer(unsigned int nsorted,
                 unsigned int nout,
                 bool streaming,
                 unsigned int outii = 0,
                 unsigned int pauseii = 0,
                 bool useAlsoVtxCoords = false);
      void initSectors(const std::vector<DetectorSector<T>>& sectors);
      void initSectors(const DetectorSector<T>& sector);
      void initRegions(const std::vector<PFInputRegion>& regions);
      void initRouting(const std::vector<Route> routes, bool validateRoutes = true);

      void reset() {
        flush();
        nevt_ = 0;
      }

      // single clock emulation
      bool step(bool newEvent, const std::vector<T>& links, std::vector<T>& out, bool mux = true);

      // single clock emulation
      bool muxonly_step(bool newEvent, bool mayFlush, const std::vector<T>& nomux_out, std::vector<T>& out);

      void destream(int iclock, const std::vector<T>& streams, std::vector<T>& out);

    private:
      unsigned int nsectors_, nregions_, nsorted_, nout_, outii_, pauseii_;
      bool streaming_, useAlsoVtxCoords_;
      std::vector<l1ct::PFRegionEmu> sectors_;
      std::vector<RegionBuffer<T>> buffers_;
      std::vector<RegionBuilder<T>> builders_;
      RegionMux<T> bigmux_;
      std::vector<Route> routes_;
      unsigned int nevt_;

      void flush() {
        for (auto& b : buffers_)
          b.flush();
      }
    };

  }  // namespace  multififo_regionizer
}  // namespace l1ct

template <typename T>
inline bool l1ct::multififo_regionizer::local_eta_phi_window(const T& t,
                                                             const l1ct::glbeta_t& etaMin,
                                                             const l1ct::glbeta_t& etaMax,
                                                             const l1ct::glbphi_t& phiMin,
                                                             const l1ct::glbphi_t& phiMax) {
  return (etaMin == etaMax) ||
         (etaMin <= t.hwEta && t.hwEta <= etaMax && ((phiMin == phiMax) || (phiMin <= t.hwPhi && t.hwPhi <= phiMax)));
}
template <>
inline bool l1ct::multififo_regionizer::local_eta_phi_window<l1ct::TkObjEmu>(const l1ct::TkObjEmu& t,
                                                                             const l1ct::glbeta_t& etaMin,
                                                                             const l1ct::glbeta_t& etaMax,
                                                                             const l1ct::glbphi_t& phiMin,
                                                                             const l1ct::glbphi_t& phiMax) {
  return (etaMin == etaMax) ||
         (etaMin <= t.hwEta && t.hwEta <= etaMax && ((phiMin == phiMax) || (phiMin <= t.hwPhi && t.hwPhi <= phiMax))) ||
         (etaMin <= t.hwVtxEta() && t.hwVtxEta() <= etaMax &&
          ((phiMin == phiMax) || (phiMin <= t.hwVtxPhi() && t.hwVtxPhi() <= phiMax)));
}
template <typename T>
void l1ct::multififo_regionizer::EtaPhiBuffer<T>::maybe_push(const T& t) {
  if ((t.hwPt != 0) && local_eta_phi_window(t, etaMin_, etaMax_, phiMin_, phiMax_)) {
    if (items_[iwrite_].size() < size_) {
      items_[iwrite_].push_back(t);
      items_[iwrite_].back().hwEta += etaShift_;
      items_[iwrite_].back().hwPhi += phiShift_;
    } else {
      // uncommenting the message below may be useful for debugging
      //dbgCout() << "WARNING: sector buffer is full for " << typeid(T).name() << ", pt = " << t.intPt()
      //          << ", eta = " << t.intEta() << ", phi = " << t.intPhi() << "\n";
    }
  }
}

template <typename T>
T l1ct::multififo_regionizer::EtaPhiBuffer<T>::pop() {
  T ret;
  ret.clear();
  if (!items_[iread_].empty()) {
    ret = items_[iread_].front();
    items_[iread_].pop_front();
  }
  return ret;
}
template <typename T>
void l1ct::multififo_regionizer::EtaPhiBuffer<T>::reset() {
  iread_ = 0;
  iwrite_ = 0;
  items_[0].clear();
  items_[1].clear();
}

#endif
