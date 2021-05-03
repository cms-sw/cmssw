#ifndef LINPUPPI_REF_H
#define LINPUPPI_REF_H

#include "../dataformats/layer1_emulator.h"

#include <vector>

namespace edm {
  class ParameterSet;
}

namespace l1ct {

  class LinPuppiEmulator {
  public:
    LinPuppiEmulator(unsigned int nTrack,
                     unsigned int nIn,
                     unsigned int nOut,
                     unsigned int dR2Min,
                     unsigned int dR2Max,
                     unsigned int iptMax,
                     unsigned int dzCut,
                     double ptSlopeNe,
                     double ptSlopePh,
                     double ptZeroNe,
                     double ptZeroPh,
                     double alphaSlope,
                     double alphaZero,
                     double alphaCrop,
                     double priorNe,
                     double priorPh,
                     pt_t ptCut)
        : nTrack_(nTrack),
          nIn_(nIn),
          nOut_(nOut),
          dR2Min_(dR2Min),
          dR2Max_(dR2Max),
          iptMax_(iptMax),
          dzCut_(dzCut),
          absEtaBins_(),
          ptSlopeNe_(1, ptSlopeNe),
          ptSlopePh_(1, ptSlopePh),
          ptZeroNe_(1, ptZeroNe),
          ptZeroPh_(1, ptZeroPh),
          alphaSlope_(1, alphaSlope),
          alphaZero_(1, alphaZero),
          alphaCrop_(1, alphaCrop),
          priorNe_(1, priorNe),
          priorPh_(1, priorPh),
          ptCut_(1, ptCut),
          debug_(false) {}

    LinPuppiEmulator(unsigned int nTrack,
                     unsigned int nIn,
                     unsigned int nOut,
                     unsigned int dR2Min,
                     unsigned int dR2Max,
                     unsigned int iptMax,
                     unsigned int dzCut,
                     glbeta_t etaCut,
                     double ptSlopeNe_0,
                     double ptSlopeNe_1,
                     double ptSlopePh_0,
                     double ptSlopePh_1,
                     double ptZeroNe_0,
                     double ptZeroNe_1,
                     double ptZeroPh_0,
                     double ptZeroPh_1,
                     double alphaSlope_0,
                     double alphaSlope_1,
                     double alphaZero_0,
                     double alphaZero_1,
                     double alphaCrop_0,
                     double alphaCrop_1,
                     double priorNe_0,
                     double priorNe_1,
                     double priorPh_0,
                     double priorPh_1,
                     pt_t ptCut_0,
                     pt_t ptCut_1);

    LinPuppiEmulator(unsigned int nTrack,
                     unsigned int nIn,
                     unsigned int nOut,
                     unsigned int dR2Min,
                     unsigned int dR2Max,
                     unsigned int iptMax,
                     unsigned int dzCut,
                     const std::vector<glbeta_t> &absEtaBins,
                     const std::vector<double> &ptSlopeNe,
                     const std::vector<double> &ptSlopePh,
                     const std::vector<double> &ptZeroNe,
                     const std::vector<double> &ptZeroPh,
                     const std::vector<double> &alphaSlope,
                     const std::vector<double> &alphaZero,
                     const std::vector<double> &alphaCrop,
                     const std::vector<double> &priorNe,
                     const std::vector<double> &priorPh,
                     const std::vector<pt_t> &ptCut)
        : nTrack_(nTrack),
          nIn_(nIn),
          nOut_(nOut),
          dR2Min_(dR2Min),
          dR2Max_(dR2Max),
          iptMax_(iptMax),
          dzCut_(dzCut),
          absEtaBins_(absEtaBins),
          ptSlopeNe_(ptSlopeNe),
          ptSlopePh_(ptSlopePh),
          ptZeroNe_(ptZeroNe),
          ptZeroPh_(ptZeroPh),
          alphaSlope_(alphaSlope),
          alphaZero_(alphaZero),
          alphaCrop_(alphaCrop),
          priorNe_(priorNe),
          priorPh_(priorPh),
          ptCut_(ptCut),
          debug_(false) {}

    LinPuppiEmulator(const edm::ParameterSet &iConfig);

    // charged
    void linpuppi_chs_ref(const PFRegionEmu &region,
                          const PVObjEmu &pv,
                          const std::vector<PFChargedObjEmu> &pfch /*[nTrack]*/,
                          std::vector<PuppiObjEmu> &outallch /*[nTrack]*/) const;

    // neutrals, in the tracker
    void linpuppi_flt(const PFRegionEmu &region,
                      const std::vector<TkObjEmu> &track /*[nTrack]*/,
                      const PVObjEmu &pv,
                      const std::vector<PFNeutralObjEmu> &pfallne /*[nIn]*/,
                      std::vector<PuppiObjEmu> &outallne_nocut /*[nIn]*/,
                      std::vector<PuppiObjEmu> &outallne /*[nIn]*/,
                      std::vector<PuppiObjEmu> &outselne /*[nOut]*/) const;
    void linpuppi_ref(const PFRegionEmu &region,
                      const std::vector<TkObjEmu> &track /*[nTrack]*/,
                      const PVObjEmu &pv,
                      const std::vector<PFNeutralObjEmu> &pfallne /*[nIn]*/,
                      std::vector<PuppiObjEmu> &outallne_nocut /*[nIn]*/,
                      std::vector<PuppiObjEmu> &outallne /*[nIn]*/,
                      std::vector<PuppiObjEmu> &outselne /*[nOut]*/) const;
    void linpuppi_ref(const PFRegionEmu &region,
                      const std::vector<TkObjEmu> &track /*[nTrack]*/,
                      const PVObjEmu &pv,
                      const std::vector<PFNeutralObjEmu> &pfallne /*[nIn]*/,
                      std::vector<PuppiObjEmu> &outselne /*[nOut]*/) const {
      std::vector<PuppiObjEmu> outallne_nocut, outallne;
      linpuppi_ref(region, track, pv, pfallne, outallne_nocut, outallne, outselne);
    }

    // neutrals, forward
    void fwdlinpuppi_ref(const PFRegionEmu &region,
                         const std::vector<HadCaloObjEmu> &caloin /*[nIn]*/,
                         std::vector<PuppiObjEmu> &outallne_nocut /*[nIn]*/,
                         std::vector<PuppiObjEmu> &outallne /*[nIn]*/,
                         std::vector<PuppiObjEmu> &outselne /*[nOut]*/) const;
    void fwdlinpuppi_flt(const PFRegionEmu &region,
                         const std::vector<HadCaloObjEmu> &caloin /*[nIn]*/,
                         std::vector<PuppiObjEmu> &outallne_nocut /*[nIn]*/,
                         std::vector<PuppiObjEmu> &outallne /*[nIn]*/,
                         std::vector<PuppiObjEmu> &outselne /*[nOut]*/) const;

    // utility
    void puppisort_and_crop_ref(unsigned int nOutMax,
                                const std::vector<PuppiObjEmu> &in,
                                std::vector<PuppiObjEmu> &out /*nOut*/) const;

    // for CMSSW
    void run(const PFInputRegion &in, const std::vector<l1ct::PVObjEmu> &pvs, OutputRegion &out) const;

    void setDebug(bool debug = true) { debug_ = debug; }

  protected:
    unsigned int nTrack_, nIn_,
        nOut_;  // nIn_, nOut refer to the calorimeter clusters or neutral PF candidates as input and as output (after sorting)
    unsigned int dR2Min_, dR2Max_, iptMax_, dzCut_;
    std::vector<glbeta_t> absEtaBins_;
    std::vector<double> ptSlopeNe_, ptSlopePh_, ptZeroNe_, ptZeroPh_;
    std::vector<double> alphaSlope_, alphaZero_, alphaCrop_;
    std::vector<double> priorNe_, priorPh_;
    std::vector<pt_t> ptCut_;

    bool debug_;

    // utility
    unsigned int find_ieta(const PFRegionEmu &region, eta_t eta) const;
    std::pair<pt_t, puppiWgt_t> sum2puppiPt_ref(uint64_t sum, pt_t pt, unsigned int ieta, bool isEM, int icand) const;
    std::pair<float, float> sum2puppiPt_flt(float sum, float pt, unsigned int ieta, bool isEM, int icand) const;
  };

}  // namespace l1ct

#endif
