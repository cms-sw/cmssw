#ifndef L1TRIGGER_PHASE2L1PARTICLEFLOW_BUFFERED_FOLDED_MULTIFIFO_REGIONZER_REF_H
#define L1TRIGGER_PHASE2L1PARTICLEFLOW_BUFFERED_FOLDED_MULTIFIFO_REGIONZER_REF_H

#include "L1Trigger/Phase2L1ParticleFlow/interface/regionizer/folded_multififo_regionizer_ref.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/dbgPrintf.h"
#include <memory>
#include <deque>

namespace l1ct {
  namespace multififo_regionizer {
    template <typename T>
    inline bool local_eta_window(const T& t, const l1ct::glbeta_t& etaMin, const l1ct::glbeta_t& etaMax);
    template <>
    inline bool local_eta_window<l1ct::TkObjEmu>(const l1ct::TkObjEmu& t,
                                                 const l1ct::glbeta_t& etaMin,
                                                 const l1ct::glbeta_t& etaMax);

    template <typename T>
    class EtaBuffer {
    public:
      EtaBuffer() {}
      EtaBuffer(unsigned int maxitems, const l1ct::glbeta_t& etaMin = 0, const l1ct::glbeta_t& etaMax = 0)
          : size_(maxitems), iwrite_(0), iread_(0), etaMin_(etaMin), etaMax_(etaMax) {}
      void maybe_push(const T& t);
      void writeNewEvent() {
        iwrite_ = 1 - iwrite_;
        items_[iwrite_].clear();
      }
      void readNewEvent() { iread_ = 1 - iread_; }
      T pop();
      unsigned int writeSize() const { return items_[iwrite_].size(); }
      unsigned int readSize() const { return items_[iread_].size(); }

    private:
      unsigned int size_, iwrite_, iread_;
      l1ct::glbeta_t etaMin_, etaMax_;
      std::deque<T> items_[2];
    };
  }  // namespace multififo_regionizer
}  // namespace l1ct
namespace l1ct {
  class BufferedFoldedMultififoRegionizerEmulator : public FoldedMultififoRegionizerEmulator {
  public:
    enum class FoldMode { EndcapEta2 };

    BufferedFoldedMultififoRegionizerEmulator(unsigned int nclocks,
                                              unsigned int ntk,
                                              unsigned int ncalo,
                                              unsigned int nem,
                                              unsigned int nmu,
                                              bool streaming,
                                              unsigned int outii,
                                              unsigned int pauseii,
                                              bool useAlsoVtxCoords);
    // note: this one will work only in CMSSW
    BufferedFoldedMultififoRegionizerEmulator(const edm::ParameterSet& iConfig);

    ~BufferedFoldedMultififoRegionizerEmulator() override;

    static edm::ParameterSetDescription getParameterSetDescription();

    void initSectorsAndRegions(const RegionizerDecodedInputs& in, const std::vector<PFInputRegion>& out) override;

    void run(const RegionizerDecodedInputs& in, std::vector<PFInputRegion>& out) override;

    void fillLinks(unsigned int iclock, std::vector<l1ct::TkObjEmu>& links, std::vector<bool>& valid);
    void fillLinks(unsigned int iclock, std::vector<l1ct::HadCaloObjEmu>& links, std::vector<bool>& valid);
    void fillLinks(unsigned int iclock, std::vector<l1ct::EmCaloObjEmu>& links, std::vector<bool>& valid);
    void fillLinks(unsigned int iclock, std::vector<l1ct::MuObjEmu>& links, std::vector<bool>& valid);

    // clock-cycle emulation
    bool step(bool newEvent,
              const std::vector<l1ct::TkObjEmu>& links_tk,
              const std::vector<l1ct::HadCaloObjEmu>& links_hadCalo,
              const std::vector<l1ct::EmCaloObjEmu>& links_emCalo,
              const std::vector<l1ct::MuObjEmu>& links_mu,
              std::vector<l1ct::TkObjEmu>& out_tk,
              std::vector<l1ct::HadCaloObjEmu>& out_hadCalo,
              std::vector<l1ct::EmCaloObjEmu>& out_emCalo,
              std::vector<l1ct::MuObjEmu>& out_mu,
              bool mux = true);

    template <typename TEmu, typename TFw>
    void toFirmware(const std::vector<TEmu>& emu, TFw fw[]) {
      for (unsigned int i = 0, n = emu.size(); i < n; ++i) {
        fw[i] = emu[i];
      }
    }

  protected:
    std::vector<l1ct::multififo_regionizer::EtaBuffer<l1ct::TkObjEmu>> tkBuffers_;
    std::vector<l1ct::multififo_regionizer::EtaBuffer<l1ct::HadCaloObjEmu>> caloBuffers_;
    std::vector<l1ct::multififo_regionizer::EtaBuffer<l1ct::MuObjEmu>> muBuffers_;

    void findEtaBounds_(const l1ct::PFRegionEmu& sec,
                        const std::vector<PFInputRegion>& reg,
                        l1ct::glbeta_t& etaMin,
                        l1ct::glbeta_t& etaMax);

    template <typename T>
    void fillLinksPosNeg_(unsigned int iclock,
                          const std::vector<l1ct::DetectorSector<T>>& secNeg,
                          const std::vector<l1ct::DetectorSector<T>>& secPos,
                          std::vector<T>& links,
                          std::vector<bool>& valid);
  };
}  // namespace l1ct

template <typename T>
inline bool l1ct::multififo_regionizer::local_eta_window(const T& t,
                                                         const l1ct::glbeta_t& etaMin,
                                                         const l1ct::glbeta_t& etaMax) {
  return (etaMin == etaMax) || (etaMin <= t.hwEta && t.hwEta <= etaMax);
}
template <>
inline bool l1ct::multififo_regionizer::local_eta_window<l1ct::TkObjEmu>(const l1ct::TkObjEmu& t,
                                                                         const l1ct::glbeta_t& etaMin,
                                                                         const l1ct::glbeta_t& etaMax) {
  return (etaMin == etaMax) || (etaMin <= t.hwEta && t.hwEta <= etaMax) ||
         (etaMin <= t.hwVtxEta() && t.hwVtxEta() <= etaMax);
}
template <typename T>
void l1ct::multififo_regionizer::EtaBuffer<T>::maybe_push(const T& t) {
  if ((t.hwPt != 0) && local_eta_window(t, etaMin_, etaMax_)) {
    if (items_[iwrite_].size() < size_) {
      items_[iwrite_].push_back(t);
    } else {
      // uncommenting the message below may be useful for debugging
      //dbgCout() << "WARNING: sector buffer is full for " << typeid(T).name() << ", pt = " << t.intPt()
      //          << ", eta = " << t.intEta() << ", phi = " << t.intPhi() << "\n";
    }
  }
}

template <typename T>
T l1ct::multififo_regionizer::EtaBuffer<T>::pop() {
  T ret;
  ret.clear();
  if (!items_[iread_].empty()) {
    ret = items_[iread_].front();
    items_[iread_].pop_front();
  }
  return ret;
}

#endif
