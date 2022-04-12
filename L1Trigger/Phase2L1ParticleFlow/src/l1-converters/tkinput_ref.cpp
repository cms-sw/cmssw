#include "L1Trigger/Phase2L1ParticleFlow/interface/l1-converters/tkinput_ref.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/dbgPrintf.h"

#include <cmath>
#include <cassert>
#include <cstdio>

#ifdef CMSSW_GIT_HASH
#include <cstdint>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace {
  l1ct::TrackInputEmulator::Region parseRegion(const std::string &str) {
    if (str == "barrel")
      return l1ct::TrackInputEmulator::Region::Barrel;
    else if (str == "endcap")
      return l1ct::TrackInputEmulator::Region::Endcap;
    else if (str == "any")
      return l1ct::TrackInputEmulator::Region::Any;
    else
      throw cms::Exception("Configuration", "TrackInputEmulator: Unsupported region '" + str + "'\n");
  }
  l1ct::TrackInputEmulator::Encoding parseEncoding(const std::string &str) {
    if (str == "stepping")
      return l1ct::TrackInputEmulator::Encoding::Stepping;
    else if (str == "biased")
      return l1ct::TrackInputEmulator::Encoding::Biased;
    else if (str == "unbised")
      return l1ct::TrackInputEmulator::Encoding::Unbiased;
    else
      throw cms::Exception("Configuration", "TrackInputEmulator: Unsupported track word encoding '" + str + "'\n");
  }
}  // namespace

l1ct::TrackInputEmulator::TrackInputEmulator(const edm::ParameterSet &iConfig)
    : TrackInputEmulator(parseRegion(iConfig.getParameter<std::string>("region")),
                         parseEncoding(iConfig.getParameter<std::string>("trackWordEncoding")),
                         iConfig.getParameter<bool>("bitwiseAccurate")) {
  if (region_ != Region::Endcap && region_ != Region::Barrel) {
    edm::LogError("TrackInputEmulator") << "region '" << iConfig.getParameter<std::string>("region")
                                        << "' is not yet supported";
  }
  debug_ = iConfig.getUntrackedParameter<bool>("debug", false);
  configPt(iConfig.getParameter<uint32_t>("ptLUTBits"));
  configEta(iConfig.getParameter<uint32_t>("etaLUTBits"),
            iConfig.getParameter<int32_t>("etaPreOffs"),
            iConfig.getParameter<uint32_t>("etaShift"),
            iConfig.getParameter<int32_t>("etaPostOffs"),
            iConfig.getParameter<bool>("etaSigned"),
            region_ == Region::Endcap);
  configPhi(iConfig.getParameter<uint32_t>("phiBits"));
  configZ0(iConfig.getParameter<uint32_t>("z0Bits"));
  if (region_ == Region::Barrel) {
    configDEtaBarrel(iConfig.getParameter<uint32_t>("dEtaBarrelBits"),
                     iConfig.getParameter<uint32_t>("dEtaBarrelZ0PreShift"),
                     iConfig.getParameter<uint32_t>("dEtaBarrelZ0PostShift"),
                     iConfig.getParameter<double>("dEtaBarrelFloatOffs"));
    configDPhiBarrel(iConfig.getParameter<uint32_t>("dPhiBarrelBits"),
                     iConfig.getParameter<uint32_t>("dPhiBarrelRInvPreShift"),
                     iConfig.getParameter<uint32_t>("dPhiBarrelRInvPostShift"),
                     iConfig.getParameter<double>("dPhiBarrelFloatOffs"));
  }
  if (region_ == Region::Endcap) {
    configDEtaHGCal(iConfig.getParameter<uint32_t>("dEtaHGCalBits"),
                    iConfig.getParameter<uint32_t>("dEtaHGCalZ0PreShift"),
                    iConfig.getParameter<uint32_t>("dEtaHGCalRInvPreShift"),
                    iConfig.getParameter<uint32_t>("dEtaHGCalLUTBits"),
                    iConfig.getParameter<uint32_t>("dEtaHGCalLUTShift"),
                    iConfig.getParameter<double>("dEtaHGCalFloatOffs"));
    configDPhiHGCal(iConfig.getParameter<uint32_t>("dPhiHGCalBits"),
                    iConfig.getParameter<uint32_t>("dPhiHGCalZ0PreShift"),
                    iConfig.getParameter<uint32_t>("dPhiHGCalZ0PostShift"),
                    iConfig.getParameter<uint32_t>("dPhiHGCalRInvShift"),
                    iConfig.getParameter<uint32_t>("dPhiHGCalTanlInvShift"),
                    iConfig.getParameter<uint32_t>("dPhiHGCalTanlLUTBits"),
                    iConfig.getParameter<double>("dPhiHGCalFloatOffs"));
  }
}

#endif

l1ct::TrackInputEmulator::TrackInputEmulator(Region region, Encoding encoding, bool bitwise)
    : region_(region),
      encoding_(encoding),
      bitwise_(bitwise),
      rInvToPt_(31199.5),
      phiScale_(0.00038349520),
      z0Scale_(0.00999469),
      dEtaBarrelParamZ0_(0.31735),
      dPhiBarrelParamC_(0.0056535),
      dEtaHGCalParamZ0_(-0.00655),
      dEtaHGCalParamRInv2C_(+0.66),
      dEtaHGCalParamRInv2ITanl1_(-0.72),
      dEtaHGCalParamRInv2ITanl2_(-0.38),
      dPhiHGCalParamZ0_(0.00171908),
      dPhiHGCalParamC_(56.5354),
      debug_(false) {}

std::pair<l1ct::TkObjEmu, bool> l1ct::TrackInputEmulator::decodeTrack(ap_uint<96> tkword,
                                                                      const l1ct::PFRegionEmu &sector,
                                                                      bool bitwise) const {
  l1ct::TkObjEmu ret;
  ret.clear();
  auto z0 = signedZ0(tkword);
  auto tanl = signedTanl(tkword);
  auto Rinv = signedRinv(tkword);
  auto phi = signedPhi(tkword);

  bool okprop = false, oksel = false;
  switch (region_) {
    case Region::Barrel:
      okprop = withinBarrel(tanl);
      break;
    case Region::Endcap:
      okprop = mayReachHGCal(tanl) && withinTracker(tanl);
      break;
    case Region::Any:
      okprop = withinTracker(tanl);
      break;
  }

  if (valid(tkword) && okprop) {
    ret.hwQuality = tkword(2, 0);
    ret.hwCharge = charge(tkword);

    if (bitwise) {
      ret.hwPt = convPt(Rinv);

      l1ct::glbeta_t vtxEta = convEta(tanl);
      l1ct::phi_t vtxPhi = convPhi(phi);

      // track propagation
      if (region_ == Region::Barrel) {
        ret.hwDEta = calcDEtaBarrel(z0, Rinv, tanl);
        ret.hwDPhi = calcDPhiBarrel(z0, Rinv, tanl);
      }

      if (region_ == Region::Endcap) {
        ret.hwDEta = calcDEtaHGCal(z0, Rinv, tanl);
        ret.hwDPhi = calcDPhiHGCal(z0, Rinv, tanl);
      }

      ret.hwEta = vtxEta - ret.hwDEta;
      ret.hwPhi = vtxPhi - ret.hwDPhi * ret.intCharge();
      ret.hwZ0 = convZ0(z0);
    } else {
      ret.hwPt = l1ct::Scales::makePtFromFloat(floatPt(Rinv));

      float fvtxEta = floatEta(tanl) / l1ct::Scales::ETAPHI_LSB;
      float fvtxPhi = floatPhi(phi) / l1ct::Scales::ETAPHI_LSB;

      // track propagation
      float fDEta = 0, fDPhi = 0;  // already in layer-1 units
      if (region_ == Region::Barrel) {
        fDEta = floatDEtaBarrel(z0, Rinv, tanl);
        fDPhi = floatDPhiBarrel(z0, Rinv, tanl);
      }

      if (region_ == Region::Endcap) {
        fDEta = floatDEtaHGCal(z0, Rinv, tanl);
        fDPhi = floatDPhiHGCal(z0, Rinv, tanl);
      }

      ret.hwDPhi = std::round(fDPhi);
      ret.hwDEta = std::round(fDEta);
      ret.hwPhi = std::round(fvtxPhi - fDPhi * ret.intCharge());
      ret.hwEta = glbeta_t(std::round(fvtxEta)) - ret.hwDEta - sector.hwEtaCenter;

      ret.hwZ0 = l1ct::Scales::makeZ0(floatZ0(z0));
    }

    oksel = ret.hwQuality != 0;
  }
  return std::make_pair(ret, oksel);
}

float l1ct::TrackInputEmulator::floatPt(ap_int<15> Rinv) const { return rInvToPt_ / std::abs(toFloat_(Rinv)); }

l1ct::pt_t l1ct::TrackInputEmulator::convPt(ap_int<15> Rinv) const {
  ap_uint<14> absRinv = (Rinv >= 0 ? ap_uint<14>(Rinv) : ap_uint<14>(-Rinv));
  unsigned int index = absRinv.to_int() >> ptLUTShift_;
  if (index >= ptLUT_.size()) {
    dbgPrintf("WARN: Rinv %d, absRinv %d, index %d, size %lu, shift %d\n",
              Rinv.to_int(),
              absRinv.to_int(),
              index,
              ptLUT_.size(),
              ptLUTShift_);
    index = ptLUT_.size() - 1;
  }
  return ptLUT_[index];
}

void l1ct::TrackInputEmulator::configPt(int lutBits) {
  ptLUTShift_ = 14 - lutBits;
  ptLUT_.resize(1 << lutBits);
  for (unsigned int u = 0, n = ptLUT_.size(); u < n; ++u) {
    int iRinv = std::round((u + 0.5) * (1 << ptLUTShift_));
    ptLUT_[u] = l1ct::Scales::makePtFromFloat(floatPt(iRinv));
  }
}

float l1ct::TrackInputEmulator::floatEta(ap_int<16> tanl) const {
  float lam = std::atan(floatTanl(tanl));
  float theta = M_PI / 2 - lam;
  return -std::log(std::tan(0.5 * theta));
}

l1ct::glbeta_t l1ct::TrackInputEmulator::convEta(ap_int<16> tanl) const {
  unsigned int index;
  if (tanlLUTSigned_) {
    index = std::max(0, std::abs(tanl.to_int()) - tanlLUTPreOffs_) >> tanlLUTShift_;
  } else {
    ap_uint<16> unsTanl = tanl(15, 0);
    index = unsTanl.to_int() >> tanlLUTShift_;
  }
  if (index >= tanlLUT_.size()) {
    dbgPrintf(
        "WARN: tanl %d, index %d, size %lu (signed %d)\n", tanl.to_int(), index, tanlLUT_.size(), int(tanlLUTSigned_));
    index = tanlLUT_.size() - 1;
  }
  int ret = tanlLUT_[index] + tanlLUTPostOffs_;
  if (tanlLUTSigned_ && tanl < 0)
    ret = -ret;
  if (debug_)
    dbgPrintf("convEta: itanl = %+8d -> index %8d, LUT %8d, ret %+8d\n", tanl.to_int(), index, tanlLUT_[index], ret);
  return ret;
}

void l1ct::TrackInputEmulator::configEta(
    int lutBits, int preOffs, int shift, int postOffs, bool lutSigned, bool endcap) {
  tanlLUTSigned_ = lutSigned;
  tanlLUTPreOffs_ = preOffs;
  tanlLUTPostOffs_ = postOffs;
  tanlLUTShift_ = shift;
  tanlLUT_.resize(1 << lutBits);
  int etaCenter = lutSigned ? l1ct::Scales::makeGlbEtaRoundEven(2.5).to_int() / 2 : 0;
  int etamin = 1, etamax = -1;
  for (unsigned int u = 0, n = tanlLUT_.size(), h = n / 2; u < n; ++u) {
    int i = (tanlLUTSigned_ || (u < h)) ? int(u) : int(u) - int(n);
    ap_int<16> tanl = std::min<int>(i * (1 << shift) + preOffs, (1 << 16) - 1);
    int eta = l1ct::Scales::makeGlbEta(floatEta(tanl)).to_int() - etaCenter - tanlLUTPostOffs_;
    bool valid = endcap ? (mayReachHGCal(tanl) && withinTracker(tanl)) : withinBarrel(tanl);
    if (valid) {
      tanlLUT_[u] = eta;
      if (etamin > etamax) {
        etamin = eta;
        etamax = eta;
      } else {
        etamin = std::min(etamin, eta);
        etamax = std::max(etamax, eta);
      }
    } else {
      tanlLUT_[u] = 0;
    }
  }
  if (debug_)
    dbgPrintf(
        "Configured with glbEtaCenter = %d, bits %d, preOffs %d, shift %d, postOffs %d, lutmin = %d, lutmax = %d\n",
        etaCenter,
        lutBits,
        preOffs,
        shift,
        postOffs,
        etamin,
        etamax);
}

float l1ct::TrackInputEmulator::floatPhi(ap_int<12> phi) const { return phiScale_ * toFloat_(phi); }

l1ct::phi_t l1ct::TrackInputEmulator::convPhi(ap_int<12> phi) const {
  int offs = phi >= 0 ? vtxPhiOffsPos_ : vtxPhiOffsNeg_;
  return (phi.to_int() * vtxPhiMult_ + offs) >> vtxPhiBitShift_;
}

void l1ct::TrackInputEmulator::configPhi(int bits) {
  float scale = phiScale_ / l1ct::Scales::ETAPHI_LSB;
  vtxPhiBitShift_ = bits;
  vtxPhiMult_ = std::round(scale * (1 << bits));
  switch (encoding_) {
    case Encoding::Stepping:
      vtxPhiOffsPos_ = std::round(+scale * 0.5 * (1 << bits) + 0.5 * (1 << bits));
      vtxPhiOffsNeg_ = std::round(-scale * 0.5 * (1 << bits) + 0.5 * (1 << bits));
      break;
    case Encoding::Biased:
      vtxPhiOffsPos_ = std::round(+scale * 0.5 * (1 << bits) + 0.5 * (1 << bits));
      vtxPhiOffsNeg_ = std::round(+scale * 0.5 * (1 << bits) + 0.5 * (1 << bits));
      break;
    case Encoding::Unbiased:
      vtxPhiOffsPos_ = (1 << (bits - 1));
      vtxPhiOffsNeg_ = (1 << (bits - 1));
      break;
  }
  if (debug_)
    dbgPrintf("Configured vtxPhi with scale %d [to_cmssw %.8f, to_l1ct %.8f, %d bits], offsets %+d (pos), %+d (neg)\n",
              vtxPhiMult_,
              phiScale_,
              scale,
              bits,
              vtxPhiOffsPos_,
              vtxPhiOffsNeg_);
}

float l1ct::TrackInputEmulator::floatZ0(ap_int<12> z0) const { return z0Scale_ * toFloat_(z0); }

l1ct::z0_t l1ct::TrackInputEmulator::convZ0(ap_int<12> z0) const {
  int offs = z0 >= 0 ? z0OffsPos_ : z0OffsNeg_;
  return (z0.to_int() * z0Mult_ + offs) >> z0BitShift_;
}

void l1ct::TrackInputEmulator::configZ0(int bits) {
  float scale = z0Scale_ / l1ct::Scales::Z0_LSB;
  z0BitShift_ = bits;
  z0Mult_ = std::round(scale * (1 << bits));
  switch (encoding_) {
    case Encoding::Stepping:
      z0OffsPos_ = std::round(+scale * 0.5 * (1 << bits) + 0.5 * (1 << bits));
      z0OffsNeg_ = std::round(-scale * 0.5 * (1 << bits) + 0.5 * (1 << bits));
      break;
    case Encoding::Biased:
      z0OffsPos_ = std::round(+scale * 0.5 * (1 << bits) + 0.5 * (1 << bits));
      z0OffsNeg_ = std::round(+scale * 0.5 * (1 << bits) + 0.5 * (1 << bits));
      break;
    case Encoding::Unbiased:
      z0OffsPos_ = (1 << (bits - 1));
      z0OffsNeg_ = (1 << (bits - 1));
      break;
  }

  if (debug_)
    dbgPrintf("Configured z0 with scale %d [to_cmssw %.8f, to_l1ct %.8f, %d bits], offsets %+d (pos), %+d (neg)\n",
              z0Mult_,
              z0Scale_,
              scale,
              bits,
              z0OffsPos_,
              z0OffsNeg_);
}

float l1ct::TrackInputEmulator::floatDEtaBarrel(ap_int<12> z0, ap_int<15> Rinv, ap_int<16> tanl) const {
  float ret = floatEta(tanl) - floatEta(tanl + z0.to_float() * dEtaBarrelParamZ0_);
  if (debug_) {
    dbgPrintf(
        "flt deta for z0 %+6d Rinv %+6d tanl %+6d:  eta(calo) %+8.2f  eta(vtx)  %+8.3f  ret  "
        "%+8.2f\n",
        z0.to_int(),
        Rinv.to_int(),
        tanl.to_int(),
        floatEta(tanl + z0.to_float() * dEtaBarrelParamZ0_),
        floatEta(tanl),
        ret);
  }
  return ret / l1ct::Scales::ETAPHI_LSB;
}

l1ct::tkdeta_t l1ct::TrackInputEmulator::calcDEtaBarrel(ap_int<12> z0, ap_int<15> Rinv, ap_int<16> tanl) const {
  int vtxEta = convEta(tanl);

  ap_uint<14> absZ0 = z0 >= 0 ? ap_uint<14>(z0) : ap_uint<14>(-z0);
  int preSum = ((absZ0 >> dEtaBarrelZ0PreShift_) * dEtaBarrelZ0_) >> dEtaBarrelZ0PostShift_;

  int caloEta = convEta(tanl + (z0 > 0 ? 1 : -1) * ((preSum + dEtaBarrelOffs_) >> dEtaBarrelBits_));

  int ret = vtxEta - caloEta;
  if (debug_) {
    dbgPrintf(
        "int deta for z0 %+6d Rinv %+6d tanl %+6d:  preSum %+8.2f  eta(calo) %+8.2f  eta(vtx)  %+8.3f  ret  "
        "%+8.2f\n",
        z0.to_int(),
        Rinv.to_int(),
        tanl.to_int(),
        preSum,
        caloEta,
        vtxEta,
        ret);
  }
  return ret;
}

//use eta LUTs
void l1ct::TrackInputEmulator::configDEtaBarrel(int dEtaBarrelBits,
                                                int dEtaBarrelZ0PreShift,
                                                int dEtaBarrelZ0PostShift,
                                                float offs) {
  dEtaBarrelBits_ = dEtaBarrelBits;

  dEtaBarrelZ0PreShift_ = dEtaBarrelZ0PreShift;
  dEtaBarrelZ0PostShift_ = dEtaBarrelZ0PostShift;
  dEtaBarrelZ0_ =
      std::round(dEtaBarrelParamZ0_ * (1 << (dEtaBarrelZ0PreShift + dEtaBarrelZ0PostShift + dEtaBarrelBits)));

  int finalShift = dEtaBarrelBits_;
  dEtaBarrelOffs_ = std::round((1 << finalShift) * (0.5 + offs));

  if (debug_)
    dbgPrintf("Configured deta with %d bits: preshift %8d  postshift %8d, offset %8d\n",
              dEtaBarrelBits,
              dEtaBarrelZ0PreShift_,
              dEtaBarrelZ0PostShift_,
              offs);

  assert(finalShift >= 0);
}

float l1ct::TrackInputEmulator::floatDPhiBarrel(ap_int<12> z0, ap_int<15> Rinv, ap_int<16> tanl) const {
  float ret = dPhiBarrelParamC_ * std::abs(Rinv.to_int());
  //ret = atan(ret / sqrt(1-ret*ret)); //use linear approx for now
  if (debug_) {
    dbgPrintf("flt dphi for z0 %+6d Rinv %+6d tanl %+6d:  Rinv/1k  %8.2f   ret  %8.2f\n",
              z0.to_int(),
              Rinv.to_int(),
              tanl.to_int(),
              std::abs(Rinv.to_int()) / 1024.0,
              ret);
  }
  return ret;
}

l1ct::tkdphi_t l1ct::TrackInputEmulator::calcDPhiBarrel(ap_int<12> z0, ap_int<15> Rinv, ap_int<16> tanl) const {
  ap_uint<14> absRinv = Rinv >= 0 ? ap_uint<14>(Rinv) : ap_uint<14>(-Rinv);
  int preSum = ((absRinv >> dPhiBarrelRInvPreShift_) * dPhiBarrelC_) >> dPhiBarrelRInvPostShift_;

  if (debug_) {
    dbgPrintf("int dphi for z0 %+6d Rinv %+6d tanl %+6d:  ret  %8.2f\n",
              z0.to_int(),
              Rinv.to_int(),
              tanl.to_int(),
              (preSum + dPhiBarrelOffs_) >> dPhiBarrelBits_);
  }

  return (preSum + dPhiBarrelOffs_) >> dPhiBarrelBits_;
}

//using DSPs
void l1ct::TrackInputEmulator::configDPhiBarrel(int dPhiBarrelBits,
                                                int dPhiBarrelRInvPreShift,
                                                int dPhiBarrelRInvPostShift,
                                                float offs) {
  dPhiBarrelBits_ = dPhiBarrelBits;

  dPhiBarrelRInvPreShift_ = dPhiBarrelRInvPreShift;
  dPhiBarrelRInvPostShift_ = dPhiBarrelRInvPostShift;
  dPhiBarrelC_ =
      std::round(dPhiBarrelParamC_ * (1 << (dPhiBarrelRInvPreShift + dPhiBarrelRInvPostShift + dPhiBarrelBits)));

  int finalShift = dPhiBarrelBits_;
  dPhiBarrelOffs_ = std::round((1 << finalShift) * (0.5 + offs));

  if (debug_)
    dbgPrintf("Configured dphi with %d bits: preshift %8d  postshift %8d, offset %8d\n",
              dPhiBarrelBits,
              dPhiBarrelRInvPreShift_,
              dPhiBarrelRInvPostShift_,
              offs);

  assert(finalShift >= 0);
}

float l1ct::TrackInputEmulator::floatDEtaHGCal(ap_int<12> z0, ap_int<15> Rinv, ap_int<16> tanl) const {
  float RinvScaled = Rinv.to_float() / (16 * 1024.0), RinvScaled2 = RinvScaled * RinvScaled;
  float invtanlScaled = (32 * 1024.0) / std::abs(tanl.to_float()), invtanlScaled2 = invtanlScaled * invtanlScaled;
  float tanlTerm = (tanl > 0 ? 1 : -1) * (dEtaHGCalParamRInv2C_ + dEtaHGCalParamRInv2ITanl1_ * invtanlScaled +
                                          dEtaHGCalParamRInv2ITanl2_ * invtanlScaled2);
  float ret = dEtaHGCalParamZ0_ * z0.to_float() + tanlTerm * RinvScaled2;
  if (debug_) {
    dbgPrintf(
        "flt deta for z0 %+6d Rinv %+6d tanl %+6d:  z0term %+8.2f  rinv2u %.4f tanlterm  %+8.3f (pre: %+8.2f)  ret  "
        "%+8.2f\n",
        z0.to_int(),
        Rinv.to_int(),
        tanl.to_int(),
        dEtaHGCalParamZ0_ * z0.to_float(),
        RinvScaled2,
        tanlTerm * RinvScaled2,
        tanlTerm,
        ret);
  }
  return ret;
}

l1ct::tkdeta_t l1ct::TrackInputEmulator::calcDEtaHGCal(ap_int<12> z0, ap_int<15> Rinv, ap_int<16> tanl) const {
  int z0Term = dEtaHGCalZ0_ * (z0 >> dEtaHGCalZ0PreShift_);

  int rinvShift = Rinv.to_int() >> dEtaHGCalRInvPreShift_, rinvShift2 = rinvShift * rinvShift;

  ap_uint<16> unsTanl = tanl(15, 0);
  unsigned int tanlIdx = (unsTanl.to_int()) >> dEtaHGCalTanlShift_;
  assert(tanlIdx < dEtaHGCalLUT_.size());
  int tanlTerm = (rinvShift2 * dEtaHGCalLUT_[tanlIdx] + dEtaHGCalTanlTermOffs_) >> dEtaHGCalTanlTermShift_;

  int ret0 = z0Term + tanlTerm + dEtaHGCalOffs_;
  if (debug_) {
    dbgPrintf(
        "int deta for z0 %+6d Rinv %+6d tanl %+6d:  z0term %+8.2f  rinv2u %.4f tanlterm  %+8.2f (pre: %+8.2f)  ret  "
        "%+8.2f\n",
        z0.to_int(),
        Rinv.to_int(),
        tanl.to_int(),
        float(z0Term) / (1 << dEtaHGCalBits_),
        float(rinvShift2) / (1 << (28 - 2 * dEtaHGCalRInvPreShift_)),
        float(tanlTerm) / (1 << dEtaHGCalBits_),
        float(dEtaHGCalLUT_[tanlIdx]) / (1 << (dEtaHGCalBits_ - dEtaHGCalLUTShift_)),
        float(ret0) / (1 << dEtaHGCalBits_));
  }
  return (ret0 + (1 << (dEtaHGCalBits_ - 1))) >> dEtaHGCalBits_;
}

void l1ct::TrackInputEmulator::configDEtaHGCal(int dEtaHGCalBits,
                                               int dEtaHGCalZ0PreShift,
                                               int dEtaHGCalRInvPreShift,
                                               int dEtaHGCalLUTBits,
                                               int dEtaHGCalLUTShift,
                                               float offs) {
  dEtaHGCalBits_ = dEtaHGCalBits;
  float scale = (1 << dEtaHGCalBits);

  dEtaHGCalZ0PreShift_ = dEtaHGCalZ0PreShift;
  dEtaHGCalZ0_ = std::round(dEtaHGCalParamZ0_ * scale * (1 << dEtaHGCalZ0PreShift));

  dEtaHGCalRInvPreShift_ = dEtaHGCalRInvPreShift;

  dEtaHGCalTanlShift_ = 16 - dEtaHGCalLUTBits;
  dEtaHGCalLUT_.resize((1 << dEtaHGCalLUTBits));
  dEtaHGCalLUTShift_ = dEtaHGCalLUTShift;

  dEtaHGCalTanlTermShift_ = 28 - 2 * dEtaHGCalRInvPreShift_ - dEtaHGCalLUTShift_;
  dEtaHGCalTanlTermOffs_ = std::round(0.5 * (1 << dEtaHGCalTanlTermShift_));
  int lutmin = 1, lutmax = -1;
  float lutScale = scale / (1 << dEtaHGCalLUTShift);
  for (unsigned int u = 0, n = dEtaHGCalLUT_.size(), h = n / 2; u < n; ++u) {
    int i = (u < h) ? int(u) : int(u) - int(n);
    float tanl = (i + 0.5) * (1 << dEtaHGCalTanlShift_);
    float sign = tanl >= 0 ? 1 : -1;
    float invtanlScaled = 32 * 1024.0 / std::abs(tanl), invtanlScaled2 = invtanlScaled * invtanlScaled;
    float term = sign * (dEtaHGCalParamRInv2C_ + dEtaHGCalParamRInv2ITanl1_ * invtanlScaled +
                         dEtaHGCalParamRInv2ITanl2_ * invtanlScaled2);
    int iterm = std::round(lutScale * term);
    bool valid = mayReachHGCal(tanl);
    if (valid) {
      dEtaHGCalLUT_[u] = iterm;
      if (lutmin > lutmax) {
        lutmin = iterm;
        lutmax = iterm;
      } else {
        lutmin = std::min(lutmin, iterm);
        lutmax = std::max(lutmax, iterm);
      }
    } else {
      dEtaHGCalLUT_[u] = 0;
    }
  }

  dEtaHGCalOffs_ = std::round(scale * offs);

  if (debug_)
    dbgPrintf(
        "Configured deta with %d bits: z0 %8d [%8.2f], lutmin = %d, lutmax = %d, lutshift %d, rinvShift %d, "
        "tanlTermShift %d\n",
        dEtaHGCalBits,
        dEtaHGCalZ0_,
        dEtaHGCalZ0_ / float(scale),
        lutmin,
        lutmax,
        dEtaHGCalLUTShift,
        dEtaHGCalRInvPreShift_,
        dEtaHGCalTanlTermShift_);
}

float l1ct::TrackInputEmulator::floatDPhiHGCal(ap_int<12> z0, ap_int<15> Rinv, ap_int<16> tanl) const {
  int dzsign = tanl >= 0 ? +1 : -1;
  float ret =
      (dPhiHGCalParamC_ - dPhiHGCalParamZ0_ * z0 * dzsign) * std::abs(Rinv.to_int()) / std::abs(tanl.to_float());
  if (debug_) {
    dbgPrintf(
        "flt dphi for z0 %+6d Rinv %+6d tanl %+6d:  preSum %+9.4f  Rinv/1k  %8.2f   1k/tanl  %8.5f   ret  %8.2f\n",
        z0.to_int(),
        Rinv.to_int(),
        tanl.to_int(),
        dPhiHGCalParamC_ - dPhiHGCalParamZ0_ * z0 * dzsign,
        std::abs(Rinv.to_int()) / 1024.0,
        1024.0 / std::abs(tanl.to_float()),
        ret);
  }
  return ret;
}

l1ct::tkdphi_t l1ct::TrackInputEmulator::calcDPhiHGCal(ap_int<12> z0, ap_int<15> Rinv, ap_int<16> tanl) const {
  int dzsign = tanl >= 0 ? +1 : -1;
  int preSum = (((z0 >> dPhiHGCalZ0PreShift_) * dPhiHGCalZ0_) >> dPhiHGCalZ0PostShift_) * dzsign + dPhiHGCalPreOffs_;

  ap_uint<14> absRinv = Rinv >= 0 ? ap_uint<14>(Rinv) : ap_uint<14>(-Rinv);
  int rinvShifted = absRinv.to_int() >> dPhiHGCalRInvShift_;

  ap_uint<15> absTanl = tanl >= 0 ? ap_uint<15>(tanl) : ap_uint<15>(-tanl);
  unsigned int tanlIdx = absTanl.to_int() >> dPhiHGCalTanlShift_;
  assert(tanlIdx < dPhiHGCalTanlLUT_.size());
  int tanlTerm = dPhiHGCalTanlLUT_[tanlIdx];

  int finalShift = dPhiHGCalBits_ + dPhiHGCalTanlInvShift_ - dPhiHGCalRInvShift_;
  if (debug_) {
    dbgPrintf(
        "int dphi for z0 %+6d Rinv %+6d tanl %+6d:  preSum %+9.4f  Rinv/1k  %8.2f   1k/tanl  %8.5f   ret  %8.2f: int "
        "preSum %8d  rivShift %8d  tanlTerm %8d\n",
        z0.to_int(),
        Rinv.to_int(),
        tanl.to_int(),
        float(preSum) / (1 << dPhiHGCalBits_),
        (rinvShifted << dPhiHGCalRInvShift_) / 1024.0,
        tanlTerm * 1024.0 / (1 << dPhiHGCalTanlInvShift_),
        float(preSum * rinvShifted * tanlTerm) / (1 << finalShift),
        preSum,
        rinvShifted,
        tanlTerm);
  }

  return (preSum * rinvShifted * tanlTerm + dPhiHGCalOffs_) >> finalShift;
}

void l1ct::TrackInputEmulator::configDPhiHGCal(int dPhiHGCalBits,
                                               int dPhiHGCalZ0PreShift,
                                               int dPhiHGCalZ0PostShift,
                                               int dPhiHGCalRInvShift,
                                               int dPhiHGCalTanlInvShift,
                                               int dPhiHGCalTanlLUTBits,
                                               float offs) {
  dPhiHGCalBits_ = dPhiHGCalBits;

  dPhiHGCalZ0PreShift_ = dPhiHGCalZ0PreShift;
  dPhiHGCalZ0PostShift_ = dPhiHGCalZ0PostShift;
  dPhiHGCalZ0_ = -std::round(dPhiHGCalParamZ0_ * (1 << (dPhiHGCalZ0PreShift + dPhiHGCalZ0PostShift + dPhiHGCalBits)));

  dPhiHGCalPreOffs_ = std::round(dPhiHGCalParamC_ * (1 << dPhiHGCalBits));

  dPhiHGCalRInvShift_ = dPhiHGCalRInvShift;

  dPhiHGCalTanlInvShift_ = dPhiHGCalTanlInvShift;
  dPhiHGCalTanlShift_ = 15 - dPhiHGCalTanlLUTBits;
  dPhiHGCalTanlLUT_.resize((1 << dPhiHGCalTanlLUTBits));
  int lutmin = 1, lutmax = -1;
  for (unsigned int u = 0, n = dPhiHGCalTanlLUT_.size(); u < n; ++u) {
    float tanl = (u + 0.5) * (1 << dPhiHGCalTanlShift_);
    int iterm = std::round((1 << dPhiHGCalTanlInvShift_) / tanl);
    bool valid = mayReachHGCal(tanl);
    if (valid) {
      dPhiHGCalTanlLUT_[u] = iterm;
      if (lutmin > lutmax) {
        lutmin = iterm;
        lutmax = iterm;
      } else {
        lutmin = std::min(lutmin, iterm);
        lutmax = std::max(lutmax, iterm);
      }
    } else {
      dPhiHGCalTanlLUT_[u] = 0;
    }
  }

  int finalShift = dPhiHGCalBits_ + dPhiHGCalTanlInvShift_ - dPhiHGCalRInvShift_;
  dPhiHGCalOffs_ = std::round((1 << finalShift) * (0.5 + offs));

  if (debug_)
    dbgPrintf(
        "Configured dphi with %d bits: z0 %8d [%8.2f], preoffs %8d [%8.2f], final shift %d, lutmin = %d, lutmax = %d\n",
        dPhiHGCalBits,
        dPhiHGCalZ0_,
        dPhiHGCalZ0_ / float(1 << (dPhiHGCalZ0PostShift + dPhiHGCalBits)),
        dPhiHGCalPreOffs_,
        dPhiHGCalPreOffs_ / float(1 << dPhiHGCalBits),
        finalShift,
        lutmin,
        lutmax);

  assert(finalShift >= 0);
}
