#ifndef PFTKEGALGO_REF_H
#define PFTKEGALGO_REF_H

#ifdef CMSSW_GIT_HASH
#include "../dataformats/layer1_emulator.h"
#include "../dataformats/egamma.h"
#include "../dataformats/pf.h"
#else
#include "../../../dataformats/layer1_emulator.h"
#include "../../../dataformats/egamma.h"
#include "../../../dataformats/pf.h"
#endif

namespace edm {
  class ParameterSet;
}

namespace l1ct {

  struct pftkegalgo_config {
    unsigned int nTRACK;
    unsigned int nEMCALO;
    unsigned int nEMCALOSEL_EGIN;
    unsigned int nEM_EGOUT;

    bool filterHwQuality;
    bool doBremRecovery;
    int caloHwQual;
    float emClusterPtMin;  // GeV
    float dEtaMaxBrem;
    float dPhiMaxBrem;

    std::vector<double> absEtaBoundaries;
    std::vector<double> dEtaValues;
    std::vector<double> dPhiValues;
    float trkQualityPtMin;  // GeV

    pftkegalgo_config(const edm::ParameterSet &iConfig);
    pftkegalgo_config(unsigned int nTrack,
                      unsigned int nEmCalo,
                      unsigned int nEmCaloSel_in,
                      unsigned int nEmOut,
                      bool filterHwQuality,
                      bool doBremRecovery,
                      int caloHwQual = 4,
                      float emClusterPtMin = 2.,
                      float dEtaMaxBrem = 0.02,
                      float dPhiMaxBrem = 0.1,
                      std::vector<double> absEtaBoundaries = {0.0, 1.5},
                      std::vector<double> dEtaValues = {0.015, 0.0174533},
                      std::vector<double> dPhiValues = {0.07, 0.07},
                      float trkQualityPtMin = 10.)
        : nTRACK(nTrack),
          nEMCALO(nEmCalo),
          nEMCALOSEL_EGIN(nEmCaloSel_in),
          nEM_EGOUT(nEmOut),
          filterHwQuality(filterHwQuality),
          doBremRecovery(doBremRecovery),
          caloHwQual(caloHwQual),
          emClusterPtMin(emClusterPtMin),
          dEtaMaxBrem(dEtaMaxBrem),
          dPhiMaxBrem(dPhiMaxBrem),
          absEtaBoundaries(std::move(absEtaBoundaries)),
          dEtaValues(std::move(dEtaValues)),
          dPhiValues(std::move(dPhiValues)),
          trkQualityPtMin(trkQualityPtMin) {}
  };

  class PFTkEGAlgoEmulator {
  public:
    PFTkEGAlgoEmulator(const pftkegalgo_config &config) : cfg(config) {}

    virtual ~PFTkEGAlgoEmulator() {}

    void toFirmware(const PFInputRegion &in, PFRegion &region, EmCaloObj calo[/*nCALO*/], TkObj track[/*nTRACK*/]) const;
    void toFirmware(const OutputRegion &out, EGIsoObj out_egphs[], EGIsoEleObj out_egeles[]) const;

    virtual void run(const PFInputRegion &in, OutputRegion &out) const;

    void setDebug(int verbose) { debug_ = verbose; }

  private:
    void link_emCalo2emCalo(const std::vector<EmCaloObjEmu> &emcalo, std::vector<int> &emCalo2emCalo) const;

    void link_emCalo2tk(const PFRegionEmu &r,
                        const std::vector<EmCaloObjEmu> &emcalo,
                        const std::vector<TkObjEmu> &track,
                        std::vector<int> &emCalo2tk) const;

    //FIXME: still needed
    float deltaPhi(float phi1, float phi2) const;

    void sel_emCalo(unsigned int nmax_sel,
                    const std::vector<EmCaloObjEmu> &emcalo,
                    std::vector<EmCaloObjEmu> &emcalo_sel) const;

    void eg_algo(const std::vector<EmCaloObjEmu> &emcalo,
                 const std::vector<TkObjEmu> &track,
                 const std::vector<int> &emCalo2emCalo,
                 const std::vector<int> &emCalo2tk,
                 std::vector<EGIsoObjEmu> &egobjs,
                 std::vector<EGIsoEleObjEmu> &egeleobjs) const;

    void addEgObjsToPF(const std::vector<EmCaloObjEmu> &emcalo,
                       const std::vector<TkObjEmu> &track,
                       const int calo_idx,
                       const int hwQual,
                       const pt_t ptCorr,
                       const int tk_idx,
                       std::vector<EGIsoObjEmu> &egobjs,
                       std::vector<EGIsoEleObjEmu> &egeleobjs) const;

    EGIsoObjEmu &addEGIsoToPF(std::vector<EGIsoObjEmu> &egobjs,
                              const EmCaloObjEmu &calo,
                              const int hwQual,
                              const pt_t ptCorr) const;

    EGIsoEleObjEmu &addEGIsoEleToPF(std::vector<EGIsoEleObjEmu> &egobjs,
                                    const EmCaloObjEmu &calo,
                                    const TkObjEmu &track,
                                    const int hwQual,
                                    const pt_t ptCorr) const;

    // FIXME: reimplemented from PFAlgoEmulatorBase
    template <typename T>
    void ptsort_ref(int nIn, int nOut, const std::vector<T> &in, std::vector<T> &out) const {
      // // std based sort
      // out = in;
      // std::sort(out.begin(), out.end(),  std::greater<T>());
      // out.resize(nOut);
      out.resize(nOut);
      for (int iout = 0; iout < nOut; ++iout) {
        out[iout].clear();
      }
      for (int it = 0; it < nIn; ++it) {
        for (int iout = 0; iout < nOut; ++iout) {
          if (in[it].hwPt >= out[iout].hwPt) {
            for (int i2 = nOut - 1; i2 > iout; --i2) {
              out[i2] = out[i2 - 1];
            }
            out[iout] = in[it];
            break;
          }
        }
      }
    }

    pftkegalgo_config cfg;
    int debug_ = 0;
  };
}  // namespace l1ct

#endif
