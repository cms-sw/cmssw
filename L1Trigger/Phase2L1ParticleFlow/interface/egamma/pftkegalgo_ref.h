#ifndef PFTKEGALGO_REF_H
#define PFTKEGALGO_REF_H

#include "DataFormats/L1TParticleFlow/interface/layer1_emulator.h"
#include "DataFormats/L1TParticleFlow/interface/egamma.h"
#include "DataFormats/L1TParticleFlow/interface/pf.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/common/inversion.h"

// FIXME: back to the old way of including conifer.h
#include "L1Trigger/Phase2L1ParticleFlow/interface/conifer.h"
// #include "conifer.h"

namespace edm {
  class ParameterSet;
  class ParameterSetDescription;
}  // namespace edm

namespace l1ct {

  struct PFTkEGAlgoEmuConfig {
    unsigned int nTRACK;
    unsigned int nTRACK_EGIN;
    unsigned int nEMCALO_EGIN;
    unsigned int nEM_EGOUT;

    bool filterHwQuality;
    bool doBremRecovery;
    bool writeBeforeBremRecovery;
    int caloHwQual;
    bool doEndcapHwQual;
    float emClusterPtMin;  // GeV
    float dEtaMaxBrem;
    float dPhiMaxBrem;

    std::vector<double> absEtaBoundaries;
    std::vector<double> dEtaValues;
    std::vector<double> dPhiValues;
    float trkQualityPtMin;  // GeV

    enum Algo {
      undefined = -1,
      elliptic = 0,
      compositeEE_v0 = 1,
      compositeEB_v0 = 2,
      compositeEE_v1 = 3,
      compositeEB_v1 = 4
    };

    Algo algorithm;
    unsigned int nCompCandPerCluster;
    bool writeEgSta;

    struct IsoParameters {
      IsoParameters(const edm::ParameterSet &);
      IsoParameters(float tkQualityPtMin, float dZ, float dRMin, float dRMax)
          : tkQualityPtMin(Scales::makePtFromFloat(tkQualityPtMin)),
            dZ(Scales::makeZ0(dZ)),
            dRMin2(Scales::makeDR2FromFloatDR(dRMin)),
            dRMax2(Scales::makeDR2FromFloatDR(dRMax)) {}
      pt_t tkQualityPtMin;
      ap_int<z0_t::width + 1> dZ;
      int dRMin2;
      int dRMax2;
      static edm::ParameterSetDescription getParameterSetDescription();
    };

    IsoParameters tkIsoParams_tkEle;
    IsoParameters tkIsoParams_tkEm;
    IsoParameters pfIsoParams_tkEle;
    IsoParameters pfIsoParams_tkEm;
    bool doTkIso;
    bool doPfIso;
    EGIsoEleObjEmu::IsoType hwIsoTypeTkEle;
    EGIsoObjEmu::IsoType hwIsoTypeTkEm;

    struct CompIDParameters {
      CompIDParameters(const edm::ParameterSet &);
      CompIDParameters(const std::vector<double> &loose_wp_bins,
                       const std::vector<double> &loose_wp,
                       const std::vector<double> &tight_wp_bins,
                       const std::vector<double> &tight_wp,
                       const std::string &model,
                       double dphi_max,
                       double deta_max);

      std::vector<double> loose_wp_bins_;
      std::vector<double> loose_wp_;
      std::vector<double> tight_wp_bins_;
      std::vector<double> tight_wp_;
      std::string conifer_model_;
      double dPhi_max_;
      double dEta_max_;

      static edm::ParameterSetDescription getParameterSetDescription();
    };

    CompIDParameters compIDparams;

    int debug = 0;

    PFTkEGAlgoEmuConfig(const edm::ParameterSet &iConfig);
    PFTkEGAlgoEmuConfig(
        unsigned int nTrack,
        unsigned int nTrack_in,
        unsigned int nEmCalo_in,
        unsigned int nEmOut,
        bool filterHwQuality,
        bool doBremRecovery,
        bool writeBeforeBremRecovery = false,
        int caloHwQual = 4,
        bool doEndcapHwQual = false,
        float emClusterPtMin = 2.,
        float dEtaMaxBrem = 0.02,
        float dPhiMaxBrem = 0.1,
        const std::vector<double> &absEtaBoundaries = {0.0, 1.5},
        const std::vector<double> &dEtaValues = {0.015, 0.01},
        const std::vector<double> &dPhiValues = {0.07, 0.07},
        float trkQualityPtMin = 10.,
        unsigned int algo = 0,
        unsigned int nCompCandPerCluster = 4,
        bool writeEgSta = false,
        const IsoParameters &tkIsoParams_tkEle = {2., 0.6, 0.03, 0.2},
        const IsoParameters &tkIsoParams_tkEm = {2., 0.6, 0.07, 0.3},
        const IsoParameters &pfIsoParams_tkEle = {1., 0.6, 0.03, 0.2},
        const IsoParameters &pfIsoParams_tkEm = {1., 0.6, 0.07, 0.3},
        bool doTkIso = true,
        bool doPfIso = false,
        EGIsoEleObjEmu::IsoType hwIsoTypeTkEle = EGIsoEleObjEmu::IsoType::TkIso,
        EGIsoObjEmu::IsoType hwIsoTypeTkEm = EGIsoObjEmu::IsoType::TkIsoPV,
        const CompIDParameters &compIDparams = {{0.}, {-4}, {0.}, {0.214844}, "compositeID.json", 0.2, 0.2},
        int debug = 0)

        : nTRACK(nTrack),
          nTRACK_EGIN(nTrack_in),
          nEMCALO_EGIN(nEmCalo_in),
          nEM_EGOUT(nEmOut),
          filterHwQuality(filterHwQuality),
          doBremRecovery(doBremRecovery),
          writeBeforeBremRecovery(writeBeforeBremRecovery),
          caloHwQual(caloHwQual),
          doEndcapHwQual(doEndcapHwQual),
          emClusterPtMin(emClusterPtMin),
          dEtaMaxBrem(dEtaMaxBrem),
          dPhiMaxBrem(dPhiMaxBrem),
          absEtaBoundaries(absEtaBoundaries),
          dEtaValues(dEtaValues),
          dPhiValues(dPhiValues),
          trkQualityPtMin(trkQualityPtMin),
          algorithm(Algo::undefined),
          nCompCandPerCluster(nCompCandPerCluster),
          writeEgSta(writeEgSta),
          tkIsoParams_tkEle(tkIsoParams_tkEle),
          tkIsoParams_tkEm(tkIsoParams_tkEm),
          pfIsoParams_tkEle(pfIsoParams_tkEle),
          pfIsoParams_tkEm(pfIsoParams_tkEm),
          doTkIso(doTkIso),
          doPfIso(doPfIso),
          hwIsoTypeTkEle(hwIsoTypeTkEle),
          hwIsoTypeTkEm(hwIsoTypeTkEm),
          compIDparams(compIDparams),
          debug(debug) {
      if (algo == 0)
        algorithm = Algo::elliptic;
      else if (algo == 1)
        algorithm = Algo::compositeEE_v0;
      else if (algo == 2)
        algorithm = Algo::compositeEB_v0;
      else if (algo == 3)
        algorithm = Algo::compositeEE_v1;
      else if (algo == 4)
        algorithm = Algo::compositeEB_v1;
      else
        throw std::invalid_argument("[PFTkEGAlgoEmuConfig]: Unknown algorithm type: " + std::to_string(algo));
    }

    static edm::ParameterSetDescription getParameterSetDescription();
  };

  struct CompositeCandidate {
    unsigned int cluster_idx;
    unsigned int track_idx;
    double dpt;  // For sorting
  };

  class TkEGEleAssociationModel {
  public:
    TkEGEleAssociationModel(const l1ct::PFTkEGAlgoEmuConfig::CompIDParameters &params, int debug);
    virtual ~TkEGEleAssociationModel() = default;

    virtual id_score_t compute_score(const CompositeCandidate &cand,
                                     const std::vector<EmCaloObjEmu> &emcalo,
                                     const std::vector<TkObjEmu> &track,
                                     const std::vector<float> additional_vars) const = 0;

    bool geometric_match(const EmCaloObjEmu &emcalo, const TkObjEmu &track) const;

    class WP {
    public:
      enum cut_type { score_cut = 0, pt_binned_cut = 1 };

      cut_type getWPtype() const { return wp_type; }
      virtual ~WP() = default;

      virtual bool apply(const id_score_t &score, const float &var) const = 0;

    protected:
      WP(cut_type wp_type) : wp_type(wp_type) {}
      cut_type wp_type;
    };

    class SimpleWP : public WP {
    public:
      id_score_t wp_value_;
      SimpleWP(id_score_t wp_value) : WP(cut_type::score_cut), wp_value_(wp_value) {}
      bool apply(const id_score_t &score, const float &var) const override { return score >= wp_value_; }
    };

    class BinnedWP1D : public WP {
      std::vector<double> bin_low_edges_;
      std::vector<id_score_t> wp_values_;

    public:
      BinnedWP1D(const std::vector<double> &bin_low_edges, const std::vector<id_score_t> &wp_values)
          : WP(cut_type::pt_binned_cut), bin_low_edges_(bin_low_edges), wp_values_(wp_values) {}

      bool apply(const id_score_t &score, const float &var) const override {
        auto it = std::upper_bound(bin_low_edges_.begin(), bin_low_edges_.end(), var);
        unsigned int bin_index = it - bin_low_edges_.begin() - 1;
        return (score > id_score_t(wp_values_[bin_index]));
      };
    };

    static std::unique_ptr<WP> createWP(const std::vector<double> &bin_low_edges,
                                        const std::vector<double> &wp_values) {
      assert(bin_low_edges.size() == wp_values.size() && "The size of bin_low_edges must match the size of wp_values.");
      assert(wp_values.size() && "The size of bin_low_edges must not be 0.");

      std::vector<id_score_t> wp_values_apf;
      wp_values_apf.reserve(wp_values.size());
      std::transform(wp_values.begin(), wp_values.end(), std::back_inserter(wp_values_apf), [](const double &val) {
        return id_score_t(val);
      });
      if (bin_low_edges.size() == 1) {
        return std::make_unique<SimpleWP>(id_score_t(wp_values_apf[0]));
      }
      return std::make_unique<BinnedWP1D>(bin_low_edges, wp_values_apf);
    }

    bool apply_wp_loose(float score, float var) const { return loose_wp_->apply(score, var); }

    bool apply_wp_tight(float score, float var) const { return tight_wp_->apply(score, var); }

    WP::cut_type loose_wp_type() const { return loose_wp_->getWPtype(); }

    WP::cut_type tight_wp_type() const { return tight_wp_->getWPtype(); }

  private:
    std::unique_ptr<WP> loose_wp_;
    std::unique_ptr<WP> tight_wp_;
    float dphi2_max_;
    float deta2_max_;

  protected:
    int debug_;
  };

  class TkEgCID_EE_v0 : public TkEGEleAssociationModel {
  public:
    TkEgCID_EE_v0(const l1ct::PFTkEGAlgoEmuConfig::CompIDParameters &params, int debug);

    id_score_t compute_score(const CompositeCandidate &cand,
                             const std::vector<EmCaloObjEmu> &emcalo,
                             const std::vector<TkObjEmu> &track,
                             const std::vector<float> additional_vars) const override;

    typedef ap_fixed<21, 12, AP_RND_CONV, AP_SAT> bdt_feature_t;
    typedef ap_fixed<12, 3, AP_RND_CONV, AP_SAT> bdt_score_t;

  private:
    conifer::BDT<bdt_feature_t, bdt_score_t, false> *model_;
  };

  class TkEgCID_EE_v1 : public TkEGEleAssociationModel {
  public:
    TkEgCID_EE_v1(const l1ct::PFTkEGAlgoEmuConfig::CompIDParameters &params, int debug);

    id_score_t compute_score(const CompositeCandidate &cand,
                             const std::vector<EmCaloObjEmu> &emcalo,
                             const std::vector<TkObjEmu> &track,
                             const std::vector<float> additional_vars) const override;

    typedef ap_fixed<30, 20, AP_RND_CONV, AP_SAT> bdt_feature_t;
    typedef ap_fixed<30, 20, AP_RND_CONV, AP_SAT> bdt_score_t;

  private:
    conifer::BDT<bdt_feature_t, bdt_score_t, false> *model_;
  };

  class TkEgCID_EB_v0 : public TkEGEleAssociationModel {
  public:
    TkEgCID_EB_v0(const l1ct::PFTkEGAlgoEmuConfig::CompIDParameters &params, int debug);

    id_score_t compute_score(const CompositeCandidate &cand,
                             const std::vector<EmCaloObjEmu> &emcalo,
                             const std::vector<TkObjEmu> &track,
                             const std::vector<float> additional_vars) const override;

    typedef ap_fixed<24, 9, AP_RND_CONV, AP_SAT> bdt_feature_t;
    typedef ap_fixed<12, 4, AP_RND_CONV, AP_SAT> bdt_score_t;

  private:
    conifer::BDT<bdt_feature_t, bdt_score_t, false> *model_;
  };

  class TkEgCID_EB_v1 : public TkEGEleAssociationModel {
  public:
    TkEgCID_EB_v1(const l1ct::PFTkEGAlgoEmuConfig::CompIDParameters &params, int debug);

    id_score_t compute_score(const CompositeCandidate &cand,
                             const std::vector<EmCaloObjEmu> &emcalo,
                             const std::vector<TkObjEmu> &track,
                             const std::vector<float> additional_vars) const override;

    typedef ap_fixed<8, 1, AP_RND_CONV, AP_SAT> bdt_feature_t;
    typedef ap_fixed<11, 4, AP_RND_CONV, AP_SAT> bdt_score_t;

  private:
    float scale(const float &x, const float &min_x, const int &bitshift, float inf = -1) const {
      return inf + (x - min_x) / pow(2, bitshift);
    }

    conifer::BDT<bdt_feature_t, bdt_score_t, false> *model_;
  };

  class PFTkEGAlgoEmulator {
  public:
    PFTkEGAlgoEmulator(const PFTkEGAlgoEmuConfig &config);

    virtual ~PFTkEGAlgoEmulator() {}

    void toFirmware(const PFInputRegion &in, PFRegion &region, EmCaloObj calo[/*nCALO*/], TkObj track[/*nTRACK*/]) const;
    void toFirmware(const OutputRegion &out, EGIsoObj out_egphs[], EGIsoEleObj out_egeles[]) const;
    void toFirmware(const PFInputRegion &in,
                    const l1ct::PVObjEmu &pvin,
                    PFRegion &region,
                    TkObj track[/*nTRACK*/],
                    PVObj &pv) const;

    void run(const PFInputRegion &in, OutputRegion &out) const;
    void runIso(const PFInputRegion &in, const std::vector<l1ct::PVObjEmu> &pvs, OutputRegion &out) const;

    void setDebug(int verbose) { debug_ = verbose; }

    bool writeEgSta() const { return cfg.writeEgSta; }

    static float deltaPhi(float phi1, float phi2);

  private:
    void link_emCalo2emCalo(const std::vector<EmCaloObjEmu> &emcalo, std::vector<int> &emCalo2emCalo) const;

    void link_emCalo2tk_elliptic(const PFRegionEmu &r,
                                 const std::vector<EmCaloObjEmu> &emcalo,
                                 const std::vector<TkObjEmu> &track,
                                 std::vector<int> &emCalo2tk) const;

    void link_emCalo2tk_composite_eb_ee(const PFRegionEmu &r,
                                        const std::vector<EmCaloObjEmu> &emcalo,
                                        const std::vector<TkObjEmu> &track,
                                        std::vector<int> &emCalo2tk,
                                        std::vector<id_score_t> &emCaloTkBdtScore) const;

    void sel_emCalo(unsigned int nmax_sel,
                    const std::vector<EmCaloObjEmu> &emcalo,
                    std::vector<EmCaloObjEmu> &emcalo_sel) const;

    void eg_algo(const PFRegionEmu &region,
                 const std::vector<EmCaloObjEmu> &emcalo,
                 const std::vector<TkObjEmu> &track,
                 const std::vector<int> &emCalo2emCalo,
                 const std::vector<int> &emCalo2tk,
                 const std::vector<id_score_t> &emCaloTkBdtScore,
                 std::vector<EGObjEmu> &egstas,
                 std::vector<EGIsoObjEmu> &egobjs,
                 std::vector<EGIsoEleObjEmu> &egeleobjs) const;

    void addEgObjsToPF(std::vector<EGObjEmu> &egstas,
                       std::vector<EGIsoObjEmu> &egobjs,
                       std::vector<EGIsoEleObjEmu> &egeleobjs,
                       const std::vector<EmCaloObjEmu> &emcalo,
                       const std::vector<TkObjEmu> &track,
                       const int calo_idx,
                       const unsigned int hwQual,
                       const pt_t ptCorr,
                       const int tk_idx,
                       const id_score_t bdtScore,
                       const std::vector<unsigned int> &components = {}) const;

    EGObjEmu &addEGStaToPF(std::vector<EGObjEmu> &egobjs,
                           const EmCaloObjEmu &calo,
                           const unsigned int hwQual,
                           const pt_t ptCorr,
                           const std::vector<unsigned int> &components) const;

    EGIsoObjEmu &addEGIsoToPF(std::vector<EGIsoObjEmu> &egobjs,
                              const EmCaloObjEmu &calo,
                              const unsigned int hwQual,
                              const pt_t ptCorr) const;

    EGIsoEleObjEmu &addEGIsoEleToPF(std::vector<EGIsoEleObjEmu> &egobjs,
                                    const EmCaloObjEmu &calo,
                                    const TkObjEmu &track,
                                    const unsigned int hwQual,
                                    const pt_t ptCorr,
                                    const id_score_t bdtScore) const;

    // FIXME: reimplemented from PFAlgoEmulatorBase
    template <typename T>
    void ptsort_ref(int nIn, int nOut, const std::vector<T> &in, std::vector<T> &out) const {
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
    int deltaR2(const T &charged, const EGIsoObjEmu &egphoton) const {
      // NOTE: we compare Tk at vertex against the calo variable...
      return dr2_int(charged.hwVtxEta(), charged.hwVtxPhi(), egphoton.hwEta, egphoton.hwPhi);
    }

    template <typename T>
    int deltaR2(const T &charged, const EGIsoEleObjEmu &egele) const {
      return dr2_int(charged.hwVtxEta(), charged.hwVtxPhi(), egele.hwVtxEta(), egele.hwVtxPhi());
    }

    int deltaR2(const PFNeutralObjEmu &neutral, const EGIsoObjEmu &egphoton) const {
      return dr2_int(neutral.hwEta, neutral.hwPhi, egphoton.hwEta, egphoton.hwPhi);
    }

    int deltaR2(const PFNeutralObjEmu &neutral, const EGIsoEleObjEmu &egele) const {
      // NOTE: we compare Tk at vertex against the calo variable...
      return dr2_int(neutral.hwEta, neutral.hwPhi, egele.hwVtxEta(), egele.hwVtxPhi());
    }

    template <typename T>
    ap_int<z0_t::width + 1> deltaZ0(const T &charged, const EGIsoObjEmu &egphoton, z0_t z0) const {
      ap_int<z0_t::width + 1> delta = charged.hwZ0 - z0;
      if (delta < 0)
        delta = -delta;
      return delta;
    }

    template <typename T>
    ap_int<z0_t::width + 1> deltaZ0(const T &charged, const EGIsoEleObjEmu &egele, z0_t z0) const {
      ap_int<z0_t::width + 1> delta = charged.hwZ0 - egele.hwZ0;
      if (delta < 0)
        delta = -delta;
      return delta;
    }

    template <typename TCH, typename TEG>
    void compute_sumPt(iso_t &sumPt,
                       iso_t &sumPtPV,
                       const std::vector<TCH> &objects,
                       unsigned int nMaxObj,
                       const TEG &egobj,
                       const PFTkEGAlgoEmuConfig::IsoParameters &params,
                       z0_t z0) const {
      for (unsigned int itk = 0; itk < std::min<unsigned>(objects.size(), nMaxObj); ++itk) {
        const auto &obj = objects[itk];

        if (obj.hwPt < params.tkQualityPtMin)
          continue;

        int dR2 = deltaR2(obj, egobj);

        if (dR2 > params.dRMin2 && dR2 < params.dRMax2) {
          sumPt += obj.hwPt;
          if (deltaZ0(obj, egobj, z0) < params.dZ) {
            sumPtPV += obj.hwPt;
          }
        }
      }
    }

    template <typename TEG>
    void compute_sumPt(iso_t &sumPt,
                       iso_t &sumPtPV,
                       const std::vector<PFNeutralObjEmu> &objects,
                       unsigned int nMaxObj,
                       const TEG &egobj,
                       const PFTkEGAlgoEmuConfig::IsoParameters &params,
                       z0_t z0) const {
      for (unsigned int itk = 0; itk < std::min<unsigned>(objects.size(), nMaxObj); ++itk) {
        const auto &obj = objects[itk];

        if (obj.hwPt < params.tkQualityPtMin)
          continue;

        int dR2 = deltaR2(obj, egobj);

        if (dR2 > params.dRMin2 && dR2 < params.dRMax2) {
          sumPt += obj.hwPt;
          // PF neutrals are not constrained by PV (since their Z0 is 0 by design)
          sumPtPV += obj.hwPt;
        }
      }
    }

    void compute_isolation(std::vector<EGIsoObjEmu> &egobjs,
                           const std::vector<TkObjEmu> &objects,
                           const PFTkEGAlgoEmuConfig::IsoParameters &params,
                           z0_t z0) const;
    void compute_isolation(std::vector<EGIsoEleObjEmu> &egobjs,
                           const std::vector<TkObjEmu> &objects,
                           const PFTkEGAlgoEmuConfig::IsoParameters &params,
                           z0_t z0) const;
    void compute_isolation(std::vector<EGIsoObjEmu> &egobjs,
                           const std::vector<PFChargedObjEmu> &charged,
                           const std::vector<PFNeutralObjEmu> &neutrals,
                           const PFTkEGAlgoEmuConfig::IsoParameters &params,
                           z0_t z0) const;
    void compute_isolation(std::vector<EGIsoEleObjEmu> &egobjs,
                           const std::vector<PFChargedObjEmu> &charged,
                           const std::vector<PFNeutralObjEmu> &neutrals,
                           const PFTkEGAlgoEmuConfig::IsoParameters &params,
                           z0_t z0) const;

    PFTkEGAlgoEmuConfig cfg;
    // Could use a std::variant
    std::unique_ptr<TkEGEleAssociationModel> tkEleModel_;

    int debug_;
  };
}  // namespace l1ct

#endif
