////////
//
// class to store the 96-bit track word produced by the L1 Track Trigger.  Intended to be inherited by L1 TTTrack.
// packing scheme given below.
//
// author: Mike Hildreth
// modified by: Alexx Perloff
// created:     April 9, 2019
// modified:    March 9, 2021
//
///////

#include "DataFormats/L1TrackTrigger/interface/TTTrack_TrackWord.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

void tttrack_trackword::infoTestDigitizationScheme(const unsigned int nBits,
                                                   const double lsb,
                                                   const double floatingPointValue,
                                                   const unsigned int digitizedSignedValue,
                                                   const double undigitizedSignedValue,
                                                   const unsigned int redigitizedSignedValue) {
  edm::LogInfo("TTTrack_TrackWord") << "testDigitizationScheme: (nBits, lsb) = (" << nBits << ", " << lsb << ")\n"
                                    << "Floating point value = " << floatingPointValue
                                    << "\tDigitized value = " << digitizedSignedValue
                                    << "\tUn-digitized value = " << undigitizedSignedValue
                                    << "\tRe-digitized value = " << redigitizedSignedValue;
}

//Constructor - turn track parameters into 96-bit word
TTTrack_TrackWord::TTTrack_TrackWord(unsigned int valid,
                                     const GlobalVector& momentum,
                                     const GlobalPoint& POCA,
                                     double rInv,
                                     double chi2RPhi,  // would be xy chisq if chi2Z is non-zero
                                     double chi2RZ,
                                     double bendChi2,
                                     unsigned int hitPattern,
                                     double mvaQuality,
                                     unsigned int mvaOther,
                                     unsigned int sector) {
  setTrackWord(valid, momentum, POCA, rInv, chi2RPhi, chi2RZ, bendChi2, hitPattern, mvaQuality, mvaOther, sector);
}

TTTrack_TrackWord::TTTrack_TrackWord(unsigned int valid,
                                     unsigned int rInv,
                                     unsigned int phi0,
                                     unsigned int tanl,
                                     unsigned int z0,
                                     unsigned int d0,
                                     unsigned int chi2RPhi,  // would be total chisq if chi2Z is zero
                                     unsigned int chi2RZ,
                                     unsigned int bendChi2,
                                     unsigned int hitPattern,
                                     unsigned int mvaQuality,
                                     unsigned int mvaOther) {
  setTrackWord(valid, rInv, phi0, tanl, z0, d0, chi2RPhi, chi2RZ, bendChi2, hitPattern, mvaQuality, mvaOther);
}

// A setter for the floating point values
void TTTrack_TrackWord::setTrackWord(unsigned int valid,
                                     const GlobalVector& momentum,
                                     const GlobalPoint& POCA,
                                     double rInv,
                                     double chi2RPhi,  // would be total chisq if chi2Z is zero
                                     double chi2RZ,
                                     double bendChi2,
                                     unsigned int hitPattern,
                                     double mvaQuality,
                                     unsigned int mvaOther,
                                     unsigned int sector) {
  // first, derive quantities to be packed
  float rPhi = localPhi(momentum.phi(), sector);  // this needs to be phi relative to the center of the sector
  float rTanl = momentum.z() / momentum.perp();
  float rZ0 = POCA.z();
  float rD0 = POCA.perp();

  // bin and convert to integers
  valid_t valid_ = valid;
  rinv_t rInv_ = digitizeSignedValue(rInv, TrackBitWidths::kRinvSize, stepRinv);
  phi_t phi0_ = digitizeSignedValue(rPhi, TrackBitWidths::kPhiSize, stepPhi0);
  tanl_t tanl_ = digitizeSignedValue(rTanl, TrackBitWidths::kTanlSize, stepTanL);
  z0_t z0_ = digitizeSignedValue(rZ0, TrackBitWidths::kZ0Size, stepZ0);
  d0_t d0_ = digitizeSignedValue(rD0, TrackBitWidths::kD0Size, stepD0);
  chi2rphi_t chi2RPhi_ = getBin(chi2RPhi, chi2RPhiBins);
  chi2rz_t chi2RZ_ = getBin(chi2RZ, chi2RZBins);
  bendChi2_t bendChi2_ = getBin(bendChi2, bendChi2Bins);
  hit_t hitPattern_ = hitPattern;
  qualityMVA_t mvaQuality_ = getBin(mvaQuality, mvaQualityBins);
  otherMVA_t mvaOther_ = mvaOther;

  // pack the track word
  //trackWord = ( mvaOther_, mvaQuality_, hitPattern_, bendChi2_, chi2RZ_, chi2RPhi_, d0_, z0_, tanl_, phi0_, rInv_, valid_ );
  setTrackWord(
      valid_, rInv_, phi0_, tanl_, z0_, d0_, chi2RPhi_, chi2RZ_, bendChi2_, hitPattern_, mvaQuality_, mvaOther_);
}

// A setter for already-digitized values
void TTTrack_TrackWord::setTrackWord(unsigned int valid,
                                     unsigned int rInv,
                                     unsigned int phi0,
                                     unsigned int tanl,
                                     unsigned int z0,
                                     unsigned int d0,
                                     unsigned int chi2RPhi,  // would be total chisq if chi2Z is zero
                                     unsigned int chi2RZ,
                                     unsigned int bendChi2,
                                     unsigned int hitPattern,
                                     unsigned int mvaQuality,
                                     unsigned int mvaOther) {
  // bin and convert to integers
  valid_t valid_ = valid;
  rinv_t rInv_ = rInv;
  phi_t phi0_ = phi0;
  tanl_t tanl_ = tanl;
  z0_t z0_ = z0;
  d0_t d0_ = d0;
  chi2rphi_t chi2RPhi_ = chi2RPhi;
  chi2rz_t chi2RZ_ = chi2RZ;
  bendChi2_t bendChi2_ = bendChi2;
  hit_t hitPattern_ = hitPattern;
  qualityMVA_t mvaQuality_ = mvaQuality;
  otherMVA_t mvaOther_ = mvaOther;

  // pack the track word
  //trackWord = ( otherMVA_t(mvaOther), qualityMVA_t(mvaQuality), hit_t(hitPattern),
  //              bendChi2_t(bendChi2), chi2rz_t(chi2RZ), chi2rphi_t(chi2RPhi),
  //              d0_t(d0), z0_t(z0), tanl_t(tanl), phi_t(phi0), rinv_t(rInv), valid_t(valid) );
  setTrackWord(
      valid_, rInv_, phi0_, tanl_, z0_, d0_, chi2RPhi_, chi2RZ_, bendChi2_, hitPattern_, mvaQuality_, mvaOther_);
}

void TTTrack_TrackWord::setTrackWord(
    ap_uint<TrackBitWidths::kValidSize> valid,
    ap_uint<TrackBitWidths::kRinvSize> rInv,
    ap_uint<TrackBitWidths::kPhiSize> phi0,
    ap_uint<TrackBitWidths::kTanlSize> tanl,
    ap_uint<TrackBitWidths::kZ0Size> z0,
    ap_uint<TrackBitWidths::kD0Size> d0,
    ap_uint<TrackBitWidths::kChi2RPhiSize> chi2RPhi,  // would be total chisq if chi2Z is zero
    ap_uint<TrackBitWidths::kChi2RZSize> chi2RZ,
    ap_uint<TrackBitWidths::kBendChi2Size> bendChi2,
    ap_uint<TrackBitWidths::kHitPatternSize> hitPattern,
    ap_uint<TrackBitWidths::kMVAQualitySize> mvaQuality,
    ap_uint<TrackBitWidths::kMVAOtherSize> mvaOther) {
  // pack the track word
  unsigned int offset = 0;
  for (unsigned int b = offset; b < (offset + TrackBitWidths::kMVAOtherSize); b++) {
    trackWord_.set(b, mvaOther[b - offset]);
  }
  offset += TrackBitWidths::kMVAOtherSize;
  for (unsigned int b = offset; b < (offset + TrackBitWidths::kMVAQualitySize); b++) {
    trackWord_.set(b, mvaQuality[b - offset]);
  }
  offset += TrackBitWidths::kMVAQualitySize;
  for (unsigned int b = offset; b < (offset + TrackBitWidths::kHitPatternSize); b++) {
    trackWord_.set(b, hitPattern[b - offset]);
  }
  offset += TrackBitWidths::kHitPatternSize;
  for (unsigned int b = offset; b < (offset + TrackBitWidths::kBendChi2Size); b++) {
    trackWord_.set(b, bendChi2[b - offset]);
  }
  offset += TrackBitWidths::kBendChi2Size;
  for (unsigned int b = offset; b < (offset + TrackBitWidths::kD0Size); b++) {
    trackWord_.set(b, d0[b - offset]);
  }
  offset += TrackBitWidths::kD0Size;
  for (unsigned int b = offset; b < (offset + TrackBitWidths::kChi2RZSize); b++) {
    trackWord_.set(b, chi2RZ[b - offset]);
  }
  offset += TrackBitWidths::kChi2RZSize;
  for (unsigned int b = offset; b < (offset + TrackBitWidths::kZ0Size); b++) {
    trackWord_.set(b, z0[b - offset]);
  }
  offset += TrackBitWidths::kZ0Size;
  for (unsigned int b = offset; b < (offset + TrackBitWidths::kTanlSize); b++) {
    trackWord_.set(b, tanl[b - offset]);
  }
  offset += TrackBitWidths::kTanlSize;
  for (unsigned int b = offset; b < (offset + TrackBitWidths::kChi2RPhiSize); b++) {
    trackWord_.set(b, chi2RPhi[b - offset]);
  }
  offset += TrackBitWidths::kChi2RPhiSize;
  for (unsigned int b = offset; b < (offset + TrackBitWidths::kPhiSize); b++) {
    trackWord_.set(b, phi0[b - offset]);
  }
  offset += TrackBitWidths::kPhiSize;
  for (unsigned int b = offset; b < (offset + TrackBitWidths::kRinvSize); b++) {
    trackWord_.set(b, rInv[b - offset]);
  }
  offset += TrackBitWidths::kRinvSize;
  for (unsigned int b = offset; b < offset + TrackBitWidths::kValidSize; b++) {
    trackWord_.set(b, valid[b - offset]);
  }
}

bool TTTrack_TrackWord::singleDigitizationSchemeTest(const double floatingPointValue,
                                                     const unsigned int nBits,
                                                     const double lsb) const {
  unsigned int digitizedSignedValue = digitizeSignedValue(floatingPointValue, nBits, lsb);
  double undigitizedSignedValue = undigitizeSignedValue(digitizedSignedValue, nBits, lsb);
  unsigned int redigitizedSignedValue = digitizeSignedValue(undigitizedSignedValue, nBits, lsb);
  tttrack_trackword::infoTestDigitizationScheme(
      nBits, lsb, floatingPointValue, digitizedSignedValue, undigitizedSignedValue, redigitizedSignedValue);
  return (std::abs(floatingPointValue - undigitizedSignedValue) <= (lsb / 2.0)) &&
         (digitizedSignedValue == redigitizedSignedValue);
}

void TTTrack_TrackWord::testDigitizationScheme() const {
  /*
  Expected output:
    testDigitizationScheme: Floating point value = -4 Digitized value = 4  Un-digitized value = -3.5
    testDigitizationScheme: Floating point value = 3  Digitized value = 3  Un-digitized value = 3.5
    testDigitizationScheme: Floating point value = -3.5 Digitized value = 9  Un-digitized value = -3.25
    testDigitizationScheme: Floating point value = 3.5  Digitized value = 7  Un-digitized value = 3.75
  */
  assert(singleDigitizationSchemeTest(-4.0, 3, 1.0));
  assert(singleDigitizationSchemeTest(3.0, 3, 1.0));
  assert(singleDigitizationSchemeTest(-3.5, 4, 0.5));
  assert(singleDigitizationSchemeTest(3.5, 4, 0.5));
}
