#include "L1Trigger/Phase2L1ParticleFlow/interface/puppi/linpuppi_ref.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/puppi/linpuppi_bits.h"
#include <cmath>
#include <algorithm>

#include "L1Trigger/Phase2L1ParticleFlow/interface/common/bitonic_hybrid_sort_ref.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/common/bitonic_sort_ref.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/dbgPrintf.h"

#ifdef CMSSW_GIT_HASH
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/allowedValues.h"
#include "FWCore/Utilities/interface/transform.h"
#include "FWCore/Utilities/interface/Exception.h"
#endif

using namespace l1ct;

l1ct::LinPuppiEmulator::LinPuppiEmulator(unsigned int nTrack,
                                         unsigned int nIn,
                                         unsigned int nOut,
                                         unsigned int nVtx,
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
                                         pt_t ptCut_1,
                                         unsigned int nFinalSort,
                                         SortAlgo finalSortAlgo)
    : nTrack_(nTrack),
      nIn_(nIn),
      nOut_(nOut),
      nVtx_(nVtx),
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
      nFinalSort_(nFinalSort ? nFinalSort : nOut),
      finalSortAlgo_(finalSortAlgo),
      debug_(false),
      fakePuppi_(false) {
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
      nVtx_(iConfig.getParameter<uint32_t>("nVtx")),
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
      nFinalSort_(iConfig.getParameter<uint32_t>("nFinalSort")),
      debug_(iConfig.getUntrackedParameter<bool>("debug", false)),
      fakePuppi_(iConfig.getParameter<bool>("fakePuppi")) {
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
  const std::string &sortAlgo = iConfig.getParameter<std::string>("finalSortAlgo");
  if (sortAlgo == "Insertion")
    finalSortAlgo_ = SortAlgo::Insertion;
  else if (sortAlgo == "BitonicRUFL")
    finalSortAlgo_ = SortAlgo::BitonicRUFL;
  else if (sortAlgo == "BitonicHLS")
    finalSortAlgo_ = SortAlgo::BitonicHLS;
  else if (sortAlgo == "Hybrid")
    finalSortAlgo_ = SortAlgo::Hybrid;
  else if (sortAlgo == "FoldedHybrid")
    finalSortAlgo_ = SortAlgo::FoldedHybrid;
  else
    throw cms::Exception("Configuration", "unsupported finalSortAlgo '" + sortAlgo + "'");
}

edm::ParameterSetDescription l1ct::LinPuppiEmulator::getParameterSetDescription() {
  edm::ParameterSetDescription description;
  description.add<uint32_t>("nTrack");
  description.add<uint32_t>("nIn");
  description.add<uint32_t>("nOut");
  description.add<uint32_t>("nVtx", 1);
  description.add<double>("dZ");
  description.add<double>("dr");
  description.add<double>("drMin");
  description.add<double>("ptMax");
  description.add<std::vector<double>>("absEtaCuts");
  description.add<std::vector<double>>("ptCut");
  description.add<std::vector<double>>("ptSlopes");
  description.add<std::vector<double>>("ptSlopesPhoton");
  description.add<std::vector<double>>("ptZeros");
  description.add<std::vector<double>>("ptZerosPhoton");
  description.add<std::vector<double>>("alphaSlopes");
  description.add<std::vector<double>>("alphaZeros");
  description.add<std::vector<double>>("alphaCrop");
  description.add<std::vector<double>>("priors");
  description.add<std::vector<double>>("priorsPhoton");
  description.add<uint32_t>("nFinalSort");
  description.ifValue(
      edm::ParameterDescription<std::string>("finalSortAlgo", "Insertion", true),
      edm::allowedValues<std::string>("Insertion", "BitonicRUFL", "BitonicHLS", "Hybrid", "FoldedHybrid"));
  description.add<bool>("fakePuppi", false);
  description.addUntracked<bool>("debug", false);
  return description;
}
#endif

void l1ct::LinPuppiEmulator::puppisort_and_crop_ref(unsigned int nOutMax,
                                                    const std::vector<l1ct::PuppiObjEmu> &in,
                                                    std::vector<l1ct::PuppiObjEmu> &out,
                                                    SortAlgo sortAlgo) {
  const unsigned int nOut = std::min<unsigned int>(nOutMax, in.size());
  out.resize(nOut);
  for (unsigned int iout = 0; iout < nOut; ++iout) {
    out[iout].clear();
  }

  if (sortAlgo == SortAlgo::Insertion) {
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
  } else if (sortAlgo == SortAlgo::BitonicRUFL) {
    bitonic_sort_and_crop_ref(in.size(), nOut, &in[0], &out[0]);
  } else if (sortAlgo == SortAlgo::BitonicHLS || sortAlgo == SortAlgo::Hybrid) {
    hybrid_bitonic_sort_and_crop_ref(in.size(), nOut, &in[0], &out[0], sortAlgo == SortAlgo::Hybrid);
  } else if (sortAlgo == SortAlgo::FoldedHybrid) {
    folded_hybrid_bitonic_sort_and_crop_ref(in.size(), nOut, &in[0], &out[0], true);
  }
}

void l1ct::LinPuppiEmulator::linpuppi_chs_ref(const PFRegionEmu &region,
                                              const std::vector<PVObjEmu> &pv,
                                              const std::vector<PFChargedObjEmu> &pfch /*[nTrack]*/,
                                              std::vector<PuppiObjEmu> &outallch /*[nTrack]*/) const {
  const unsigned int nTrack = std::min<unsigned int>(nTrack_, pfch.size());
  outallch.resize(nTrack);
  for (unsigned int i = 0; i < nTrack; ++i) {
    int pZ0 = pfch[i].hwZ0;
    int z0diff = -99999;
    for (unsigned int j = 0; j < nVtx_; ++j) {
      if (j < pv.size()) {
        int pZ0Diff = pZ0 - pv[j].hwZ0;
        if (std::abs(z0diff) > std::abs(pZ0Diff))
          z0diff = pZ0Diff;
      }
    }
    bool accept = pfch[i].hwPt != 0;
    if (!fakePuppi_)
      accept = accept && region.isFiducial(pfch[i]) && (std::abs(z0diff) <= int(dzCut_) || pfch[i].hwId.isMuon());
    if (accept) {
      outallch[i].fill(region, pfch[i]);
      if (fakePuppi_) {                           // overwrite Dxy & TkQuality with debug information
        outallch[i].setHwDxy(dxy_t(pv[0].hwZ0));  ///hack to get this to work
        outallch[i].setHwTkQuality(region.isFiducial(pfch[i]) ? 1 : 0);
      }
      if (debug_ && pfch[i].hwPt > 0)
        dbgPrintf("ref candidate %02u pt %7.2f pid %1d   vz %+6d  dz %+6d (cut %5d), fid %1d -> pass, packed %s\n",
                  i,
                  pfch[i].floatPt(),
                  pfch[i].intId(),
                  int(pfch[i].hwZ0),
                  z0diff,
                  dzCut_,
                  region.isFiducial(pfch[i]),
                  outallch[i].pack().to_string(16).c_str());
    } else {
      outallch[i].clear();
      if (debug_ && pfch[i].hwPt > 0)
        dbgPrintf("ref candidate %02u pt %7.2f pid %1d   vz %+6d  dz %+6d (cut %5d), fid %1d -> fail\n",
                  i,
                  pfch[i].floatPt(),
                  pfch[i].intId(),
                  int(pfch[i].hwZ0),
                  z0diff,
                  dzCut_,
                  region.isFiducial(pfch[i]));
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
    /*if (in <= 3) dbgPrintf("ref [%d]:  x2a(sum = %9lu): logarg = %9lu, sumterm = %9d, table[logarg] = %9d, ret pre-crop = %9d\n", 
          in, sum, logarg, 
          alphaSlope * int((std::log2(LINPUPPI_pt2DR2_scale) - sum_bitShift)*(1 << alpha_bits) + 0.5) - alphaSlope * alphaZero,
          alphaSlope * int(std::log2(float(logarg))*(1 << alpha_bits)), 
          x2a_lut); */
  } else {
    //if (in <= 3) dbgPrintf("ref [%d]:  x2a(sum = %9lu): logarg = %9lu, ret pre-crop = %9d\n",
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
    dbgPrintf(
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
        //dbgPrintf("ref term [%2d,%2d]: dr = %8d  pt2_shift = %8lu  term = %12lu\n", in, it, dr2, std::min<uint64_t>(pt2 >> 5, PTMAX2 >> 5), term);
        assert(uint64_t(PTMAX2 << (sum_bitShift - 5)) / (dR2Min_ >> 5) <= (1 << 25));
        assert(term < (1 << 25));
        sum += term;
        //dbgPrintf("    pT cand %5.1f    pT item %5.1f    dR = %.3f   term = %.1f [dbl] = %lu [int]\n",
        //            caloin[in].floatPt(), caloin[it].floatPt(), std::sqrt(dr2*LINPUPPI_DR2LSB),
        //            double(std::min<uint64_t>(pt2 >> 5, 131071)<<15)/double(std::max<int>(dr2,dR2Min_) >> 5),
        //            term);
      }
    }
    unsigned int ieta = find_ieta(region, caloin[in].hwEta);
    std::pair<pt_t, puppiWgt_t> ptAndW = sum2puppiPt_ref(sum, caloin[in].hwPt, ieta, caloin[in].hwIsEM(), in);

    outallne_nocut[in].fill(region, caloin[in], ptAndW.first, ptAndW.second);
    if (region.isFiducial(caloin[in]) && outallne_nocut[in].hwPt >= ptCut_[ieta]) {
      outallne[in] = outallne_nocut[in];
    }
  }
  puppisort_and_crop_ref(nOut_, outallne, outselne);
}

void l1ct::LinPuppiEmulator::linpuppi_ref(const PFRegionEmu &region,
                                          const std::vector<TkObjEmu> &track /*[nTrack]*/,
                                          const std::vector<PVObjEmu> &pv, /*[nVtx]*/
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

      int pZMin = 99999;
      for (unsigned int v = 0; v < nVtx_; ++v) {
        if (v < pv.size()) {
          int ppZMin = std::abs(int(track[it].hwZ0 - pv[v].hwZ0));
          if (pZMin > ppZMin)
            pZMin = ppZMin;
        }
      }
      if (std::abs(pZMin) > int(dzCut_))
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
        //dbgPrintf("ref term [%2d,%2d]: dr = %8d  pt2_shift = %8lu  term = %12lu\n", in, it, dr2, std::min<uint64_t>(pt2 >> 5, PTMAX2 >> 5), term);
        assert(uint64_t(PTMAX2 << (sum_bitShift - 5)) / (dR2Min_ >> 5) <= (1 << 25));
        assert(term < (1 << 25));
        sum += term;
        //dbgPrintf("    pT cand %5.1f    pT item %5.1f    dR = %.3f   term = %.1f [dbl] = %lu [int]\n",
        //            pfallne[in].floatPt(), track[it].floatPt(), std::sqrt(dr2*LINPUPPI_DR2LSB),
        //            double(std::min<uint64_t>(pt2 >> 5, 131071)<<15)/double(std::max<int>(dr2,dR2Min_) >> 5),
        //            term);
      }
    }

    unsigned int ieta = find_ieta(region, pfallne[in].hwEta);
    bool isEM = (pfallne[in].hwId.isPhoton());
    std::pair<pt_t, puppiWgt_t> ptAndW = sum2puppiPt_ref(sum, pfallne[in].hwPt, ieta, isEM, in);
    if (!fakePuppi_) {
      outallne_nocut[in].fill(region, pfallne[in], ptAndW.first, ptAndW.second);
      if (region.isFiducial(pfallne[in]) && outallne_nocut[in].hwPt >= ptCut_[ieta]) {
        outallne[in] = outallne_nocut[in];
      }
    } else {  // fakePuppi: keep the full candidate, but set the Puppi weight and some debug info into it
      outallne_nocut[in].fill(region, pfallne[in], pfallne[in].hwPt, ptAndW.second);
      outallne_nocut[in].hwData[9] = region.isFiducial(pfallne[in]);
      outallne_nocut[in].hwData(20, 10) = ptAndW.first(10, 0);
      outallne[in] = outallne_nocut[in];
    }
    if (debug_ && pfallne[in].hwPt > 0 && outallne_nocut[in].hwPt > 0) {
      dbgPrintf("ref candidate %02u pt %7.2f  -> puppi pt %7.2f, fiducial %1d, packed %s\n",
                in,
                pfallne[in].floatPt(),
                outallne_nocut[in].floatPt(),
                int(region.isFiducial(pfallne[in])),
                outallne_nocut[in].pack().to_string(16).c_str());
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
    dbgPrintf(
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
    std::pair<float, float> ptAndW = sum2puppiPt_flt(sum, caloin[in].floatPt(), ieta, caloin[in].hwIsEM(), in);
    outallne_nocut[in].fill(region, caloin[in], Scales::makePtFromFloat(ptAndW.first), int(ptAndW.second * 256));
    if (region.isFiducial(caloin[in]) && outallne_nocut[in].hwPt >= ptCut_[ieta]) {
      outallne[in] = outallne_nocut[in];
    }
  }

  puppisort_and_crop_ref(nOut_, outallne, outselne);
}

void l1ct::LinPuppiEmulator::linpuppi_flt(const PFRegionEmu &region,
                                          const std::vector<TkObjEmu> &track /*[nTrack]*/,
                                          const std::vector<PVObjEmu> &pv,
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

      int pZMin = 99999;
      for (unsigned int v = 0; v < nVtx_; ++v) {
        if (v < pv.size()) {
          int ppZMin = std::abs(int(track[it].hwZ0 - pv[v].hwZ0));
          if (pZMin > ppZMin)
            pZMin = ppZMin;
        }
      }
      if (std::abs(pZMin) > int(dzCut_))
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
  if (debug_) {
    dbgPrintf("\nWill run LinPuppi in region eta %+5.2f, phi %+5.2f, pv0 int Z %+d\n",
              in.region.floatEtaCenter(),
              in.region.floatPhiCenter(),
              pvs.front().hwZ0.to_int());
  }
  if (std::abs(in.region.floatEtaCenter()) < 2.5) {  // within tracker
    std::vector<PuppiObjEmu> outallch, outallne_nocut, outallne, outselne;
    linpuppi_chs_ref(in.region, pvs, out.pfcharged, outallch);
    linpuppi_ref(in.region, in.track, pvs, out.pfneutral, outallne_nocut, outallne, outselne);
    // ensure proper sizes of the vectors, to get accurate sorting wrt firmware
    const std::vector<PuppiObjEmu> &ne = (nOut_ == nIn_ ? outallne : outselne);
    unsigned int nch = outallch.size(), nne = ne.size(), i;
    outallch.resize(nTrack_ + nOut_);
    for (i = nch; i < nTrack_; ++i)
      outallch[i].clear();
    for (unsigned int j = 0; j < nne; ++i, ++j)
      outallch[i] = ne[j];
    for (; i < nTrack_ + nOut_; ++i)
      outallch[i].clear();
    puppisort_and_crop_ref(nFinalSort_, outallch, out.puppi, finalSortAlgo_);
    // trim if needed
    while (!out.puppi.empty() && out.puppi.back().hwPt == 0)
      out.puppi.pop_back();
    out.puppi.shrink_to_fit();
  } else {  // forward
    std::vector<PuppiObjEmu> outallne_nocut, outallne;
    fwdlinpuppi_ref(in.region, in.hadcalo, outallne_nocut, outallne, out.puppi);
  }
}
