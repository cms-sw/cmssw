#include "linpuppi_ref.h"
#include <cmath>
#include <algorithm>

#ifdef CMSSW_GIT_HASH
#include "linpuppi_bits.h"
#else
#include "firmware/linpuppi_bits.h"
#endif

#ifdef CMSSW_GIT_HASH
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/transform.h"
#include "FWCore/Utilities/interface/Exception.h"
#endif

using namespace l1ct;

l1ct::LinPuppiEmulator::LinPuppiEmulator(unsigned int nTrack,
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
                                         pt_t ptCut_1)
    : nTrack_(nTrack),
      nIn_(nIn),
      nOut_(nOut),
      dR2Min_(dR2Min),
      dR2Max_(dR2Max),
      iptMax_(iptMax),
      dzCut_(dzCut),
      absEtaBins_(1, etaCut),
      ptSlopeNe_(2),
      ptSlopePh_(2),
      ptZeroNe_(2),
      ptZeroPh_(2),
      alphaSlope_(2),
      alphaZero_(2),
      alphaCrop_(2),
      priorNe_(2),
      priorPh_(2),
      ptCut_(2),
      debug_(false) {
  ptSlopeNe_[0] = ptSlopeNe_0;
  ptSlopeNe_[1] = ptSlopeNe_1;
  ptSlopePh_[0] = ptSlopePh_0;
  ptSlopePh_[1] = ptSlopePh_1;
  ptZeroNe_[0] = ptZeroNe_0;
  ptZeroNe_[1] = ptZeroNe_1;
  ptZeroPh_[0] = ptZeroPh_0;
  ptZeroPh_[1] = ptZeroPh_1;
  alphaSlope_[0] = alphaSlope_0;
  alphaSlope_[1] = alphaSlope_1;
  alphaZero_[0] = alphaZero_0;
  alphaZero_[1] = alphaZero_1;
  alphaCrop_[0] = alphaCrop_0;
  alphaCrop_[1] = alphaCrop_1;
  priorNe_[0] = priorNe_0;
  priorNe_[1] = priorNe_1;
  priorPh_[0] = priorPh_0;
  priorPh_[1] = priorPh_1;
  ptCut_[0] = ptCut_0;
  ptCut_[1] = ptCut_1;
}

#ifdef CMSSW_GIT_HASH
l1ct::LinPuppiEmulator::LinPuppiEmulator(const edm::ParameterSet &iConfig)
    : nTrack_(iConfig.getParameter<uint32_t>("nTrack")),
      nIn_(iConfig.getParameter<uint32_t>("nIn")),
      nOut_(iConfig.getParameter<uint32_t>("nOut")),
      dR2Min_(l1ct::Scales::makeDR2FromFloatDR(iConfig.getParameter<double>("drMin"))),
      dR2Max_(l1ct::Scales::makeDR2FromFloatDR(iConfig.getParameter<double>("dr"))),
      iptMax_(l1ct::Scales::intPt(l1ct::Scales::makePtFromFloat(iConfig.getParameter<double>("ptMax")))),
      dzCut_(l1ct::Scales::makeZ0(iConfig.getParameter<double>("dZ"))),
      absEtaBins_(
          edm::vector_transform(iConfig.getParameter<std::vector<double>>("absEtaCuts"), l1ct::Scales::makeGlbEta)),
      ptSlopeNe_(iConfig.getParameter<std::vector<double>>("ptSlopes")),
      ptSlopePh_(iConfig.getParameter<std::vector<double>>("ptSlopesPhoton")),
      ptZeroNe_(iConfig.getParameter<std::vector<double>>("ptZeros")),
      ptZeroPh_(iConfig.getParameter<std::vector<double>>("ptZerosPhoton")),
      alphaSlope_(iConfig.getParameter<std::vector<double>>("alphaSlopes")),
      alphaZero_(iConfig.getParameter<std::vector<double>>("alphaZeros")),
      alphaCrop_(iConfig.getParameter<std::vector<double>>("alphaCrop")),
      priorNe_(iConfig.getParameter<std::vector<double>>("priors")),
      priorPh_(iConfig.getParameter<std::vector<double>>("priorsPhoton")),
      ptCut_(edm::vector_transform(iConfig.getParameter<std::vector<double>>("ptCut"), l1ct::Scales::makePtFromFloat)),
      debug_(iConfig.getUntrackedParameter<bool>("debug", false)) {
  if (absEtaBins_.size() + 1 != ptSlopeNe_.size())
    throw cms::Exception("Configuration", "size mismatch for ptSlopes parameter");
  if (absEtaBins_.size() + 1 != ptSlopePh_.size())
    throw cms::Exception("Configuration", "size mismatch for ptSlopesPhoton parameter");
  if (absEtaBins_.size() + 1 != ptZeroPh_.size())
    throw cms::Exception("Configuration", "size mismatch for ptZeros parameter");
  if (absEtaBins_.size() + 1 != ptZeroNe_.size())
    throw cms::Exception("Configuration", "size mismatch for ptZerosPhotons parameter");
  if (absEtaBins_.size() + 1 != priorPh_.size())
    throw cms::Exception("Configuration", "size mismatch for priors parameter");
  if (absEtaBins_.size() + 1 != priorNe_.size())
    throw cms::Exception("Configuration", "size mismatch for priorsPhotons parameter");
  if (absEtaBins_.size() + 1 != alphaSlope_.size())
    throw cms::Exception("Configuration", "size mismatch for alphaSlope parameter");
  if (absEtaBins_.size() + 1 != alphaZero_.size())
    throw cms::Exception("Configuration", "size mismatch for alphaZero parameter");
  if (absEtaBins_.size() + 1 != alphaCrop_.size())
    throw cms::Exception("Configuration", "size mismatch for alphaCrop parameter");
  if (absEtaBins_.size() + 1 != ptCut_.size())
    throw cms::Exception("Configuration", "size mismatch for ptCut parameter");
}
#endif

void l1ct::LinPuppiEmulator::puppisort_and_crop_ref(unsigned int nOutMax,
                                                    const std::vector<PuppiObjEmu> &in,
                                                    std::vector<PuppiObjEmu> &out) const {
  const unsigned int nOut = std::min<unsigned int>(nOutMax, in.size());
  out.resize(nOut);
  for (unsigned int iout = 0; iout < nOut; ++iout) {
    out[iout].clear();
  }

  for (unsigned int it = 0, nIn = in.size(); it < nIn; ++it) {
    for (int iout = int(nOut) - 1; iout >= 0; --iout) {
      if (out[iout].hwPt <= in[it].hwPt) {
        if (iout == 0 || out[iout - 1].hwPt > in[it].hwPt) {
          out[iout] = in[it];
        } else {
          out[iout] = out[iout - 1];
        }
      }
    }
  }
}

void l1ct::LinPuppiEmulator::linpuppi_chs_ref(const PFRegionEmu &region,
                                              const PVObjEmu &pv,
                                              const std::vector<PFChargedObjEmu> &pfch /*[nTrack]*/,
                                              std::vector<PuppiObjEmu> &outallch /*[nTrack]*/) const {
  const unsigned int nTrack = std::min<unsigned int>(nTrack_, pfch.size());
  outallch.resize(nTrack);
  for (unsigned int i = 0; i < nTrack; ++i) {
    int z0diff = pfch[i].hwZ0 - pv.hwZ0;
    if (pfch[i].hwPt != 0 && region.isFiducial(pfch[i]) && (std::abs(z0diff) <= int(dzCut_) || pfch[i].hwId.isMuon())) {
      outallch[i].fill(region, pfch[i]);
      if (debug_ && pfch[i].hwPt > 0)
        printf("ref candidate %02u pt %7.2f pid %1d   vz %+6d  dz %+6d (cut %5d) -> pass\n",
               i,
               pfch[i].floatPt(),
               pfch[i].intId(),
               int(pfch[i].hwZ0),
               z0diff,
               dzCut_);
    } else {
      outallch[i].clear();
      if (debug_ && pfch[i].hwPt > 0)
        printf("ref candidate %02u pt %7.2f pid %1d   vz %+6d  dz %+6d (cut %5d) -> fail\n",
               i,
               pfch[i].floatPt(),
               pfch[i].intId(),
               int(pfch[i].hwZ0),
               z0diff,
               dzCut_);
    }
  }
}

unsigned int l1ct::LinPuppiEmulator::find_ieta(const PFRegionEmu &region, eta_t eta) const {
  int n = absEtaBins_.size();
  glbeta_t abseta = region.hwGlbEta(eta);
  if (abseta < 0)
    abseta = -abseta;
  for (int i = 0; i < n; ++i) {
    if (abseta <= absEtaBins_[i])
      return i;
  }
  return n;
}

std::pair<pt_t, puppiWgt_t> l1ct::LinPuppiEmulator::sum2puppiPt_ref(
    uint64_t sum, pt_t pt, unsigned int ieta, bool isEM, int icand) const {
  const int sum_bitShift = LINPUPPI_sum_bitShift;
  const int x2_bits = LINPUPPI_x2_bits;                  // decimal bits the discriminator values
  const int alpha_bits = LINPUPPI_alpha_bits;            // decimal bits of the alpha values
  const int alphaSlope_bits = LINPUPPI_alphaSlope_bits;  // decimal bits of the alphaSlope values
  const int ptSlope_bits = LINPUPPI_ptSlope_bits;        // decimal bits of the ptSlope values
  const int weight_bits = LINPUPPI_weight_bits;

  const int ptSlopeNe = ptSlopeNe_[ieta] * (1 << ptSlope_bits);
  const int ptSlopePh = ptSlopePh_[ieta] * (1 << ptSlope_bits);
  const int ptZeroNe = ptZeroNe_[ieta] / LINPUPPI_ptLSB;  // in pt scale
  const int ptZeroPh = ptZeroPh_[ieta] / LINPUPPI_ptLSB;  // in pt scale
  const int alphaCrop = alphaCrop_[ieta] * (1 << x2_bits);
  const int alphaSlopeNe =
      alphaSlope_[ieta] * std::log(2.) *
      (1 << alphaSlope_bits);  // we put a log(2) here since we compute alpha as log2(sum) instead of ln(sum)
  const int alphaSlopePh = alphaSlope_[ieta] * std::log(2.) * (1 << alphaSlope_bits);
  const int alphaZeroNe = alphaZero_[ieta] / std::log(2.) * (1 << alpha_bits);
  const int alphaZeroPh = alphaZero_[ieta] / std::log(2.) * (1 << alpha_bits);
  const int priorNe = priorNe_[ieta] * (1 << x2_bits);
  const int priorPh = priorPh_[ieta] * (1 << x2_bits);

  // -- simplest version
  //int alpha = sum > 0 ? int(std::log2(float(sum) * LINPUPPI_pt2DR2_scale / (1<<sum_bitShift)) * (1 << alpha_bits) + 0.5) :  0;
  // -- re-written bringing terms out of the log
  //int alpha = sum > 0 ? int(std::log2(float(sum))*(1 << alpha_bits) + (std::log2(LINPUPPI_pt2DR2_scale) - sum_bitShift)*(1 << alpha_bits) + 0.5 ) :  0;
  // -- re-written for a LUT implementation of the log2
  const int log2lut_bits = 10;
  int alpha = 0;
  uint64_t logarg = sum;
  if (logarg > 0) {
    alpha = int((std::log2(LINPUPPI_pt2DR2_scale) - sum_bitShift) * (1 << alpha_bits) + 0.5);
    while (logarg >= (1 << log2lut_bits)) {
      logarg = logarg >> 1;
      alpha += (1 << alpha_bits);
    }
    alpha += int(
        std::log2(float(logarg)) *
        (1
         << alpha_bits));  // the maximum value of this term is log2lut_bits * (1 << alpha_bits) ~ 10*16 = 160 => fits in ap_uint<4+alpha_bits>
  }
  int alphaZero = (isEM ? alphaZeroPh : alphaZeroNe);
  int alphaSlope = (isEM ? alphaSlopePh : alphaSlopeNe);
  int x2a = std::min(std::max(alphaSlope * (alpha - alphaZero) >> (alphaSlope_bits + alpha_bits - x2_bits), -alphaCrop),
                     alphaCrop);

  // -- re-written to fit in a single LUT
  int x2a_lut = -alphaSlope * alphaZero;
  logarg = sum;
  if (logarg > 0) {
    x2a_lut += alphaSlope * int((std::log2(LINPUPPI_pt2DR2_scale) - sum_bitShift) * (1 << alpha_bits) + 0.5);
    while (logarg >= (1 << log2lut_bits)) {
      logarg = logarg >> 1;
      x2a_lut += alphaSlope * (1 << alpha_bits);
    }
    x2a_lut += alphaSlope * int(std::log2(float(logarg)) * (1 << alpha_bits));
    /*if (in <= 3) printf("ref [%d]:  x2a(sum = %9lu): logarg = %9lu, sumterm = %9d, table[logarg] = %9d, ret pre-crop = %9d\n", 
          in, sum, logarg, 
          alphaSlope * int((std::log2(LINPUPPI_pt2DR2_scale) - sum_bitShift)*(1 << alpha_bits) + 0.5) - alphaSlope * alphaZero,
          alphaSlope * int(std::log2(float(logarg))*(1 << alpha_bits)), 
          x2a_lut); */
  } else {
    //if (in <= 3) printf("ref [%d]:  x2a(sum = %9lu): logarg = %9lu, ret pre-crop = %9d\n",
    //        in, sum, logarg, x2a_lut);
  }
  x2a_lut = std::min(std::max(x2a_lut >> (alphaSlope_bits + alpha_bits - x2_bits), -alphaCrop), alphaCrop);
  assert(x2a_lut == x2a);

  int ptZero = (isEM ? ptZeroPh : ptZeroNe);
  int ptSlope = (isEM ? ptSlopePh : ptSlopeNe);
  int x2pt = ptSlope * (Scales::ptToInt(pt) - ptZero) >> (ptSlope_bits + 2 - x2_bits);

  int prior = (isEM ? priorPh : priorNe);

  int x2 = x2a + x2pt - prior;

  int weight =
      std::min<int>(1.0 / (1.0 + std::exp(-float(x2) / (1 << x2_bits))) * (1 << weight_bits) + 0.5, (1 << weight_bits));

  pt_t ptPuppi = Scales::makePt((Scales::ptToInt(pt) * weight) >> weight_bits);

  if (debug_)
    printf(
        "ref candidate %02d pt %7.2f  em %1d  ieta %1d: alpha %+7.2f   x2a %+5d = %+7.3f  x2pt %+5d = %+7.3f   x2 %+5d "
        "= %+7.3f  --> weight %4d = %.4f  puppi pt %7.2f\n",
        icand,
        Scales::floatPt(pt),
        int(isEM),
        ieta,
        std::max<float>(alpha / float(1 << alpha_bits) * std::log(2.), -99.99f),
        x2a,
        x2a / float(1 << x2_bits),
        x2pt,
        x2pt / float(1 << x2_bits),
        x2,
        x2 / float(1 << x2_bits),
        weight,
        weight / float(1 << weight_bits),
        Scales::floatPt(ptPuppi));

  return std::make_pair(ptPuppi, puppiWgt_t(weight));
}

void l1ct::LinPuppiEmulator::fwdlinpuppi_ref(const PFRegionEmu &region,
                                             const std::vector<HadCaloObjEmu> &caloin /*[nIn]*/,
                                             std::vector<PuppiObjEmu> &outallne_nocut /*[nIn]*/,
                                             std::vector<PuppiObjEmu> &outallne /*[nIn]*/,
                                             std::vector<PuppiObjEmu> &outselne /*[nOut]*/) const {
  const unsigned int nIn = std::min<unsigned int>(nIn_, caloin.size());
  const int PTMAX2 = (iptMax_ * iptMax_);

  const int sum_bitShift = LINPUPPI_sum_bitShift;

  outallne_nocut.resize(nIn);
  outallne.resize(nIn);
  for (unsigned int in = 0; in < nIn; ++in) {
    outallne_nocut[in].clear();
    outallne[in].clear();
    if (caloin[in].hwPt == 0)
      continue;
    uint64_t sum = 0;  // 2 ^ sum_bitShift times (int pt^2)/(int dr2)
    for (unsigned int it = 0; it < nIn; ++it) {
      if (it == in || caloin[it].hwPt == 0)
        continue;
      unsigned int dr2 = dr2_int(
          caloin[it].hwEta, caloin[it].hwPhi, caloin[in].hwEta, caloin[in].hwPhi);  // if dr is inside puppi cone
      if (dr2 <= dR2Max_) {
        ap_uint<9> dr2short = (dr2 >= dR2Min_ ? dr2 : dR2Min_) >> 5;  // reduce precision to make divide LUT cheaper
        uint64_t pt2 = Scales::ptToInt(caloin[it].hwPt) * Scales::ptToInt(caloin[it].hwPt);
        uint64_t term = std::min<uint64_t>(pt2 >> 5, PTMAX2 >> 5) * ((1 << sum_bitShift) / int(dr2short));
        //      dr2short >= (dR2Min_ >> 5) = 2
        //      num <= (PTMAX2 >> 5) << sum_bitShift = (2^11) << 15 = 2^26
        //      ==> term <= 2^25
        //printf("ref term [%2d,%2d]: dr = %8d  pt2_shift = %8lu  term = %12lu\n", in, it, dr2, std::min<uint64_t>(pt2 >> 5, PTMAX2 >> 5), term);
        assert(uint64_t(PTMAX2 << (sum_bitShift - 5)) / (dR2Min_ >> 5) <= (1 << 25));
        assert(term < (1 << 25));
        sum += term;
        //printf("    pT cand %5.1f    pT item %5.1f    dR = %.3f   term = %.1f [dbl] = %lu [int]\n",
        //            caloin[in].floatPt(), caloin[it].floatPt(), std::sqrt(dr2*LINPUPPI_DR2LSB),
        //            double(std::min<uint64_t>(pt2 >> 5, 131071)<<15)/double(std::max<int>(dr2,dR2Min_) >> 5),
        //            term);
      }
    }
    unsigned int ieta = find_ieta(region, caloin[in].hwEta);
    std::pair<pt_t, puppiWgt_t> ptAndW = sum2puppiPt_ref(sum, caloin[in].hwPt, ieta, caloin[in].hwIsEM, in);

    outallne_nocut[in].fill(region, caloin[in], ptAndW.first, ptAndW.second);
    if (region.isFiducial(caloin[in]) && outallne_nocut[in].hwPt >= ptCut_[ieta]) {
      outallne[in] = outallne_nocut[in];
    }
  }
  puppisort_and_crop_ref(nOut_, outallne, outselne);
}

void l1ct::LinPuppiEmulator::linpuppi_ref(const PFRegionEmu &region,
                                          const std::vector<TkObjEmu> &track /*[nTrack]*/,
                                          const PVObjEmu &pv,
                                          const std::vector<PFNeutralObjEmu> &pfallne /*[nIn]*/,
                                          std::vector<PuppiObjEmu> &outallne_nocut /*[nIn]*/,
                                          std::vector<PuppiObjEmu> &outallne /*[nIn]*/,
                                          std::vector<PuppiObjEmu> &outselne /*[nOut]*/) const {
  const unsigned int nIn = std::min<unsigned>(nIn_, pfallne.size());
  const unsigned int nTrack = std::min<unsigned int>(nTrack_, track.size());
  const int PTMAX2 = (iptMax_ * iptMax_);

  const int sum_bitShift = LINPUPPI_sum_bitShift;

  outallne_nocut.resize(nIn);
  outallne.resize(nIn);
  for (unsigned int in = 0; in < nIn; ++in) {
    outallne_nocut[in].clear();
    outallne[in].clear();
    if (pfallne[in].hwPt == 0)
      continue;
    uint64_t sum = 0;  // 2 ^ sum_bitShift times (int pt^2)/(int dr2)
    for (unsigned int it = 0; it < nTrack; ++it) {
      if (track[it].hwPt == 0)
        continue;
      if (std::abs(int(track[it].hwZ0 - pv.hwZ0)) > int(dzCut_))
        continue;
      unsigned int dr2 = dr2_int(
          pfallne[in].hwEta, pfallne[in].hwPhi, track[it].hwEta, track[it].hwPhi);  // if dr is inside puppi cone
      if (dr2 <= dR2Max_) {
        ap_uint<9> dr2short = (dr2 >= dR2Min_ ? dr2 : dR2Min_) >> 5;  // reduce precision to make divide LUT cheaper
        uint64_t pt2 = Scales::ptToInt(track[it].hwPt) * Scales::ptToInt(track[it].hwPt);
        uint64_t term = std::min<uint64_t>(pt2 >> 5, PTMAX2 >> 5) * ((1 << sum_bitShift) / int(dr2short));
        //      dr2short >= (dR2Min_ >> 5) = 2
        //      num <= (PTMAX2 >> 5) << sum_bitShift = (2^11) << 15 = 2^26
        //      ==> term <= 2^25
        //printf("ref term [%2d,%2d]: dr = %8d  pt2_shift = %8lu  term = %12lu\n", in, it, dr2, std::min<uint64_t>(pt2 >> 5, PTMAX2 >> 5), term);
        assert(uint64_t(PTMAX2 << (sum_bitShift - 5)) / (dR2Min_ >> 5) <= (1 << 25));
        assert(term < (1 << 25));
        sum += term;
        //printf("    pT cand %5.1f    pT item %5.1f    dR = %.3f   term = %.1f [dbl] = %lu [int]\n",
        //            pfallne[in].floatPt(), track[it].floatPt(), std::sqrt(dr2*LINPUPPI_DR2LSB),
        //            double(std::min<uint64_t>(pt2 >> 5, 131071)<<15)/double(std::max<int>(dr2,dR2Min_) >> 5),
        //            term);
      }
    }

    unsigned int ieta = find_ieta(region, pfallne[in].hwEta);
    bool isEM = (pfallne[in].hwId.isPhoton());
    std::pair<pt_t, puppiWgt_t> ptAndW = sum2puppiPt_ref(sum, pfallne[in].hwPt, ieta, isEM, in);
    outallne_nocut[in].fill(region, pfallne[in], ptAndW.first, ptAndW.second);
    if (region.isFiducial(pfallne[in]) && outallne_nocut[in].hwPt >= ptCut_[ieta]) {
      outallne[in] = outallne_nocut[in];
    }
  }
  puppisort_and_crop_ref(nOut_, outallne, outselne);
}

std::pair<float, float> l1ct::LinPuppiEmulator::sum2puppiPt_flt(
    float sum, float pt, unsigned int ieta, bool isEM, int icand) const {
  float alphaZero = alphaZero_[ieta], alphaSlope = alphaSlope_[ieta], alphaCrop = alphaCrop_[ieta];
  float alpha = sum > 0 ? std::log(sum) : -9e9;
  float x2a = std::min(std::max(alphaSlope * (alpha - alphaZero), -alphaCrop), alphaCrop);

  float ptZero = (isEM ? ptZeroPh_[ieta] : ptZeroNe_[ieta]);
  float ptSlope = (isEM ? ptSlopePh_[ieta] : ptSlopeNe_[ieta]);
  float x2pt = ptSlope * (pt - ptZero);

  float prior = (isEM ? priorPh_[ieta] : priorNe_[ieta]);

  float x2 = x2a + x2pt - prior;

  float weight = 1.0 / (1.0 + std::exp(-x2));

  float puppiPt = pt * weight;
  if (debug_)
    printf(
        "flt candidate %02d pt %7.2f  em %1d  ieta %1d: alpha %+7.2f   x2a         %+7.3f  x2pt         %+7.3f   x2    "
        "     %+7.3f  --> weight        %.4f  puppi pt %7.2f\n",
        icand,
        pt,
        int(isEM),
        ieta,
        std::max(alpha, -99.99f),
        x2a,
        x2pt,
        x2,
        weight,
        puppiPt);

  return std::make_pair(puppiPt, weight);
}

void l1ct::LinPuppiEmulator::fwdlinpuppi_flt(const PFRegionEmu &region,
                                             const std::vector<HadCaloObjEmu> &caloin /*[nIn]*/,
                                             std::vector<PuppiObjEmu> &outallne_nocut /*[nIn]*/,
                                             std::vector<PuppiObjEmu> &outallne /*[nIn]*/,
                                             std::vector<PuppiObjEmu> &outselne /*[nOut]*/) const {
  const unsigned int nIn = std::min<unsigned int>(nIn_, caloin.size());
  const float f_ptMax = Scales::floatPt(Scales::makePt(iptMax_));

  outallne_nocut.resize(nIn);
  outallne.resize(nIn);
  for (unsigned int in = 0; in < nIn; ++in) {
    outallne_nocut[in].clear();
    outallne[in].clear();
    if (caloin[in].hwPt == 0)
      continue;
    float sum = 0;
    for (unsigned int it = 0; it < nIn; ++it) {
      if (it == in || caloin[it].hwPt == 0)
        continue;
      unsigned int dr2 = dr2_int(
          caloin[it].hwEta, caloin[it].hwPhi, caloin[in].hwEta, caloin[in].hwPhi);  // if dr is inside puppi cone
      if (dr2 <= dR2Max_) {
        sum += std::pow(std::min<float>(caloin[it].floatPt(), f_ptMax), 2) /
               (std::max<int>(dr2, dR2Min_) * LINPUPPI_DR2LSB);
      }
    }

    unsigned int ieta = find_ieta(region, caloin[in].hwEta);
    std::pair<float, float> ptAndW = sum2puppiPt_flt(sum, caloin[in].floatPt(), ieta, caloin[in].hwIsEM, in);
    outallne_nocut[in].fill(region, caloin[in], Scales::makePtFromFloat(ptAndW.first), int(ptAndW.second * 256));
    if (region.isFiducial(caloin[in]) && outallne_nocut[in].hwPt >= ptCut_[ieta]) {
      outallne[in] = outallne_nocut[in];
    }
  }

  puppisort_and_crop_ref(nOut_, outallne, outselne);
}

void l1ct::LinPuppiEmulator::linpuppi_flt(const PFRegionEmu &region,
                                          const std::vector<TkObjEmu> &track /*[nTrack]*/,
                                          const PVObjEmu &pv,
                                          const std::vector<PFNeutralObjEmu> &pfallne /*[nIn]*/,
                                          std::vector<PuppiObjEmu> &outallne_nocut /*[nIn]*/,
                                          std::vector<PuppiObjEmu> &outallne /*[nIn]*/,
                                          std::vector<PuppiObjEmu> &outselne /*[nOut]*/) const {
  const unsigned int nIn = std::min<unsigned>(nIn_, pfallne.size());
  const unsigned int nTrack = std::min<unsigned int>(nTrack_, track.size());
  const float f_ptMax = Scales::floatPt(Scales::makePt(iptMax_));

  outallne_nocut.resize(nIn);
  outallne.resize(nIn);
  for (unsigned int in = 0; in < nIn; ++in) {
    outallne_nocut[in].clear();
    outallne[in].clear();
    if (pfallne[in].hwPt == 0)
      continue;
    float sum = 0;
    for (unsigned int it = 0; it < nTrack; ++it) {
      if (track[it].hwPt == 0)
        continue;
      if (std::abs(int(track[it].hwZ0 - pv.hwZ0)) > int(dzCut_))
        continue;
      unsigned int dr2 = dr2_int(
          pfallne[in].hwEta, pfallne[in].hwPhi, track[it].hwEta, track[it].hwPhi);  // if dr is inside puppi cone
      if (dr2 <= dR2Max_) {
        sum += std::pow(std::min<float>(track[it].floatPt(), f_ptMax), 2) /
               (std::max<int>(dr2, dR2Min_) * LINPUPPI_DR2LSB);
      }
    }
    unsigned int ieta = find_ieta(region, pfallne[in].hwEta);
    bool isEM = pfallne[in].hwId.isPhoton();
    std::pair<float, float> ptAndW = sum2puppiPt_flt(sum, pfallne[in].floatPt(), ieta, isEM, in);
    outallne_nocut[in].fill(region, pfallne[in], Scales::makePtFromFloat(ptAndW.first), int(ptAndW.second * 256));
    if (region.isFiducial(pfallne[in]) && outallne_nocut[in].hwPt >= ptCut_[ieta]) {
      outallne[in] = outallne_nocut[in];
    }
  }
  puppisort_and_crop_ref(nOut_, outallne, outselne);
}

void l1ct::LinPuppiEmulator::run(const PFInputRegion &in,
                                 const std::vector<l1ct::PVObjEmu> &pvs,
                                 OutputRegion &out) const {
  if (std::abs(in.region.floatEtaCenter()) < 2.5) {  // within tracker
    std::vector<PuppiObjEmu> outallch, outallne_nocut, outallne, outselne;
    linpuppi_chs_ref(in.region, pvs.front(), out.pfcharged, outallch);
    linpuppi_ref(in.region, in.track, pvs.front(), out.pfneutral, outallne_nocut, outallne, outselne);
    outallch.insert(outallch.end(), outselne.begin(), outselne.end());
    puppisort_and_crop_ref(nOut_, outallch, out.puppi);
  } else {  // forward
    std::vector<PuppiObjEmu> outallne_nocut, outallne;
    fwdlinpuppi_ref(in.region, in.hadcalo, outallne_nocut, outallne, out.puppi);
  }
}
