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

  struct PFTkEGAlgoEmuConfig {
    unsigned int nTRACK;
    unsigned int nEMCALO;
    unsigned int nEMCALOSEL_EGIN;
    unsigned int nEM_EGOUT;

    bool filterHwQuality;
    bool doBremRecovery;
    bool writeBeforeBremRecovery;
    int caloHwQual;
    float emClusterPtMin;  // GeV
    float dEtaMaxBrem;
    float dPhiMaxBrem;

    std::vector<double> absEtaBoundaries;
    std::vector<double> dEtaValues;
    std::vector<double> dPhiValues;
    float trkQualityPtMin;  // GeV
    bool writeEgSta;

    struct IsoParameters {
      IsoParameters(const edm::ParameterSet &);
      IsoParameters(float tkQualityPtMin, float dZ, float dRMin, float dRMax)
          : tkQualityPtMin(tkQualityPtMin), dZ(dZ), dRMin2(dRMin * dRMin), dRMax2(dRMax * dRMax) {}
      float tkQualityPtMin;
      float dZ;
      float dRMin2;
      float dRMax2;
    };

    IsoParameters tkIsoParams_tkEle;
    IsoParameters tkIsoParams_tkEm;
    IsoParameters pfIsoParams_tkEle;
    IsoParameters pfIsoParams_tkEm;
    bool doTkIso;
    bool doPfIso;
    EGIsoEleObjEmu::IsoType hwIsoTypeTkEle;
    EGIsoObjEmu::IsoType hwIsoTypeTkEm;
    int debug = 0;

    PFTkEGAlgoEmuConfig(const edm::ParameterSet &iConfig);
    PFTkEGAlgoEmuConfig(unsigned int nTrack,
                        unsigned int nEmCalo,
                        unsigned int nEmCaloSel_in,
                        unsigned int nEmOut,
                        bool filterHwQuality,
                        bool doBremRecovery,
                        bool writeBeforeBremRecovery = false,
                        int caloHwQual = 4,
                        float emClusterPtMin = 2.,
                        float dEtaMaxBrem = 0.02,
                        float dPhiMaxBrem = 0.1,
                        const std::vector<double> &absEtaBoundaries = {0.0, 1.5},
                        const std::vector<double> &dEtaValues = {0.015, 0.0174533},
                        const std::vector<double> &dPhiValues = {0.07, 0.07},
                        float trkQualityPtMin = 10.,
                        bool writeEgSta = false,
                        const IsoParameters &tkIsoParams_tkEle = {2., 0.6, 0.03, 0.2},
                        const IsoParameters &tkIsoParams_tkEm = {2., 0.6, 0.07, 0.3},
                        const IsoParameters &pfIsoParams_tkEle = {1., 0.6, 0.03, 0.2},
                        const IsoParameters &pfIsoParams_tkEm = {1., 0.6, 0.07, 0.3},
                        bool doTkIso = true,
                        bool doPfIso = true,
                        EGIsoEleObjEmu::IsoType hwIsoTypeTkEle = EGIsoEleObjEmu::IsoType::TkIso,
                        EGIsoObjEmu::IsoType hwIsoTypeTkEm = EGIsoObjEmu::IsoType::TkIsoPV)

        : nTRACK(nTrack),
          nEMCALO(nEmCalo),
          nEMCALOSEL_EGIN(nEmCaloSel_in),
          nEM_EGOUT(nEmOut),
          filterHwQuality(filterHwQuality),
          doBremRecovery(doBremRecovery),
          writeBeforeBremRecovery(writeBeforeBremRecovery),
          caloHwQual(caloHwQual),
          emClusterPtMin(emClusterPtMin),
          dEtaMaxBrem(dEtaMaxBrem),
          dPhiMaxBrem(dPhiMaxBrem),
          absEtaBoundaries(std::move(absEtaBoundaries)),
          dEtaValues(std::move(dEtaValues)),
          dPhiValues(std::move(dPhiValues)),
          trkQualityPtMin(trkQualityPtMin),
          writeEgSta(writeEgSta),
          tkIsoParams_tkEle(tkIsoParams_tkEle),
          tkIsoParams_tkEm(tkIsoParams_tkEm),
          pfIsoParams_tkEle(pfIsoParams_tkEle),
          pfIsoParams_tkEm(pfIsoParams_tkEm),
          doTkIso(doTkIso),
          doPfIso(doPfIso),
          hwIsoTypeTkEle(hwIsoTypeTkEle),
          hwIsoTypeTkEm(hwIsoTypeTkEm) {}
  };

  class PFTkEGAlgoEmulator {
  public:
    PFTkEGAlgoEmulator(const PFTkEGAlgoEmuConfig &config) : cfg(config), debug_(cfg.debug) {}

    virtual ~PFTkEGAlgoEmulator() {}

    void toFirmware(const PFInputRegion &in, PFRegion &region, EmCaloObj calo[/*nCALO*/], TkObj track[/*nTRACK*/]) const;
    void toFirmware(const OutputRegion &out, EGIsoObj out_egphs[], EGIsoEleObj out_egeles[]) const;

    void run(const PFInputRegion &in, OutputRegion &out) const;
    void runIso(const PFInputRegion &in, const std::vector<l1ct::PVObjEmu> &pvs, OutputRegion &out) const;

    void setDebug(int verbose) { debug_ = verbose; }

    bool writeEgSta() const { return cfg.writeEgSta; }

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
                 std::vector<EGObjEmu> &egstas,
                 std::vector<EGIsoObjEmu> &egobjs,
                 std::vector<EGIsoEleObjEmu> &egeleobjs) const;

    void addEgObjsToPF(std::vector<EGObjEmu> &egstas,
                       std::vector<EGIsoObjEmu> &egobjs,
                       std::vector<EGIsoEleObjEmu> &egeleobjs,
                       const std::vector<EmCaloObjEmu> &emcalo,
                       const std::vector<TkObjEmu> &track,
                       const int calo_idx,
                       const int hwQual,
                       const pt_t ptCorr,
                       const int tk_idx,
                       const std::vector<unsigned int> &components = {}) const;

    EGObjEmu &addEGStaToPF(std::vector<EGObjEmu> &egobjs,
                           const EmCaloObjEmu &calo,
                           const int hwQual,
                           const pt_t ptCorr,
                           const std::vector<unsigned int> &components) const;

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

    template <typename T>
    float deltaR2(const T &charged, const EGIsoObjEmu &egphoton) const {
      // NOTE: we compare Tk at vertex against the calo variable...
      float d_phi = deltaPhi(charged.floatVtxPhi(), egphoton.floatPhi());
      float d_eta = charged.floatVtxEta() - egphoton.floatEta();
      return d_phi * d_phi + d_eta * d_eta;
    }

    template <typename T>
    float deltaR2(const T &charged, const EGIsoEleObjEmu &egele) const {
      float d_phi = deltaPhi(charged.floatVtxPhi(), egele.floatVtxPhi());
      float d_eta = charged.floatVtxEta() - egele.floatVtxEta();
      return d_phi * d_phi + d_eta * d_eta;
    }

    float deltaR2(const PFNeutralObjEmu &neutral, const EGIsoObjEmu &egphoton) const {
      float d_phi = deltaPhi(neutral.floatPhi(), egphoton.floatPhi());
      float d_eta = neutral.floatEta() - egphoton.floatEta();
      return d_phi * d_phi + d_eta * d_eta;
    }

    float deltaR2(const PFNeutralObjEmu &neutral, const EGIsoEleObjEmu &egele) const {
      // NOTE: we compare Tk at vertex against the calo variable...
      float d_phi = deltaPhi(neutral.floatPhi(), egele.floatVtxPhi());
      float d_eta = neutral.floatEta() - egele.floatVtxEta();
      return d_phi * d_phi + d_eta * d_eta;
    }

    template <typename T>
    float deltaZ0(const T &charged, const EGIsoObjEmu &egphoton, float z0) const {
      return std::abs(charged.floatZ0() - z0);
    }

    template <typename T>
    float deltaZ0(const T &charged, const EGIsoEleObjEmu &egele, float z0) const {
      return std::abs(charged.floatZ0() - egele.floatZ0());
    }

    template <typename TCH, typename TEG>
    void compute_sumPt(float &sumPt,
                       float &sumPtPV,
                       const std::vector<TCH> &objects,
                       const TEG &egobj,
                       const PFTkEGAlgoEmuConfig::IsoParameters &params,
                       const float z0) const {
      for (int itk = 0, ntk = objects.size(); itk < ntk; ++itk) {
        const auto &obj = objects[itk];

        if (obj.floatPt() < params.tkQualityPtMin)
          continue;

        float dR2 = deltaR2(obj, egobj);

        if (dR2 > params.dRMin2 && dR2 < params.dRMax2) {
          sumPt += obj.floatPt();
          if (deltaZ0(obj, egobj, z0) < params.dZ)
            sumPtPV += obj.floatPt();
        }
      }
    }

    template <typename TEG>
    void compute_sumPt(float &sumPt,
                       float &sumPtPV,
                       const std::vector<PFNeutralObjEmu> &objects,
                       const TEG &egobj,
                       const PFTkEGAlgoEmuConfig::IsoParameters &params,
                       const float z0) const {
      for (int itk = 0, ntk = objects.size(); itk < ntk; ++itk) {
        const auto &obj = objects[itk];

        if (obj.floatPt() < params.tkQualityPtMin)
          continue;

        float dR2 = deltaR2(obj, egobj);

        if (dR2 > params.dRMin2 && dR2 < params.dRMax2) {
          sumPt += obj.floatPt();
          // PF neutrals are not constrained by PV (since their Z0 is 0 by design)
          sumPtPV += obj.floatPt();
        }
      }
    }

    void compute_isolation(std::vector<EGIsoObjEmu> &egobjs,
                           const std::vector<TkObjEmu> &objects,
                           const PFTkEGAlgoEmuConfig::IsoParameters &params,
                           const float z0) const;
    void compute_isolation(std::vector<EGIsoEleObjEmu> &egobjs,
                           const std::vector<TkObjEmu> &objects,
                           const PFTkEGAlgoEmuConfig::IsoParameters &params,
                           const float z0) const;
    void compute_isolation(std::vector<EGIsoObjEmu> &egobjs,
                           const std::vector<PFChargedObjEmu> &charged,
                           const std::vector<PFNeutralObjEmu> &neutrals,
                           const PFTkEGAlgoEmuConfig::IsoParameters &params,
                           const float z0) const;
    void compute_isolation(std::vector<EGIsoEleObjEmu> &egobjs,
                           const std::vector<PFChargedObjEmu> &charged,
                           const std::vector<PFNeutralObjEmu> &neutrals,
                           const PFTkEGAlgoEmuConfig::IsoParameters &params,
                           const float z0) const;

    PFTkEGAlgoEmuConfig cfg;
    int debug_;
  };
}  // namespace l1ct

#endif
