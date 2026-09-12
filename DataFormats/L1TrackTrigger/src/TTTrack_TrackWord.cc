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

//Constructor - turn track parameters into 96-bit word
TTTrack_TrackWord::TTTrack_TrackWord(unsigned int valid,
                                     const GlobalVector& momentum,
                                     const GlobalPoint& POCA,
                                     double rInv,
                                     double chi2RPhiRed,  // normalized to dof
                                     double chi2RZRed,
                                     double bendChi2Red,
                                     unsigned int hitPattern,
                                     double mvaQuality,
                                     double mvaOther,
                                     unsigned int sector) {
  setTrackWord(
      valid, momentum, POCA, rInv, chi2RPhiRed, chi2RZRed, bendChi2Red, hitPattern, mvaQuality, mvaOther, sector);
}

TTTrack_TrackWord::TTTrack_TrackWord(unsigned int valid,
                                     unsigned int rInv,
                                     unsigned int localPhi,  // relative to centre of phi nonant
                                     unsigned int tanl,
                                     unsigned int z0,
                                     unsigned int d0,
                                     unsigned int chi2RPhiRed,  // normalized to dof
                                     unsigned int chi2RZRed,
                                     unsigned int bendChi2Red,
                                     unsigned int hitPattern,
                                     unsigned int mvaQuality,
                                     unsigned int mvaOther) {
  setTrackWord(
      valid, rInv, localPhi, tanl, z0, d0, chi2RPhiRed, chi2RZRed, bendChi2Red, hitPattern, mvaQuality, mvaOther);
}

// A setter for the floating point values
void TTTrack_TrackWord::setTrackWord(unsigned int valid,
                                     const GlobalVector& momentum,
                                     const GlobalPoint& POCA,
                                     double rInv,
                                     double chi2RPhiRed,  // normalized to dof
                                     double chi2RZRed,
                                     double bendChi2Red,
                                     unsigned int hitPattern,
                                     double mvaQuality,
                                     double mvaOther,
                                     unsigned int sector) {
  // first, derive quantities to be packed
  float rLocalPhi = localPhi(momentum.phi(), sector);  // phi relative to the center of the sector
  float rTanl = momentum.z() / momentum.perp();
  float rZ0 = POCA.z();
  float rD0 = POCA.x() * sin(momentum.phi()) - POCA.y() * cos(momentum.phi());

  // bin and convert to integers
  valid_t valid_HLS = valid;
  rinv_t rInv_HLS = digitizeSignedValue(rInv, TrackBitWidths::kRinvSize, stepRinv);
  phi_t phi0_HLS = digitizeSignedValue(rLocalPhi, TrackBitWidths::kPhiSize, stepPhi0);
  tanl_t tanl_HLS = digitizeSignedValue(rTanl, TrackBitWidths::kTanlSize, stepTanL);
  z0_t z0_HLS = digitizeSignedValue(rZ0, TrackBitWidths::kZ0Size, stepZ0);
  d0_t d0_HLS = digitizeSignedValue(rD0, TrackBitWidths::kD0Size, stepD0);
  chi2rphi_t chi2RPhiRed_HLS = getBin(chi2RPhiRed, chi2RPhiBins);
  chi2rz_t chi2RZRed_HLS = getBin(chi2RZRed, chi2RZBins);
  bendChi2_t bendChi2Red_HLS = getBin(bendChi2Red, bendChi2Bins);
  hit_t hitPattern_HLS = hitPattern;
  qualityMVA_t mvaQuality_HLS = getBin(mvaQuality, tqMVABins);
  otherMVA_t mvaOther_HLS = (unsigned int)(mvaOther);

  // pack the track word
  setTrackWord(valid_HLS,
               rInv_HLS,
               phi0_HLS,
               tanl_HLS,
               z0_HLS,
               d0_HLS,
               chi2RPhiRed_HLS,
               chi2RZRed_HLS,
               bendChi2Red_HLS,
               hitPattern_HLS,
               mvaQuality_HLS,
               mvaOther_HLS);
}

// A setter for already-digitized values
void TTTrack_TrackWord::setTrackWord(unsigned int valid,
                                     unsigned int rInv,
                                     unsigned int localPhi,  // relative to centre of phi nonant
                                     unsigned int tanl,
                                     unsigned int z0,
                                     unsigned int d0,
                                     unsigned int chi2RPhiRed,  // normalized to dof
                                     unsigned int chi2RZRed,
                                     unsigned int bendChi2Red,
                                     unsigned int hitPattern,
                                     unsigned int mvaQuality,
                                     unsigned int mvaOther) {
  // bin and convert to integers
  valid_t valid_HLS = valid;
  rinv_t rInv_HLS = rInv;
  phi_t localPhi_HLS = localPhi;
  tanl_t tanl_HLS = tanl;
  z0_t z0_HLS = z0;
  d0_t d0_HLS = d0;
  chi2rphi_t chi2RPhiRed_HLS = chi2RPhiRed;
  chi2rz_t chi2RZRed_HLS = chi2RZRed;
  bendChi2_t bendChi2Red_HLS = bendChi2Red;
  hit_t hitPattern_HLS = hitPattern;
  qualityMVA_t mvaQuality_HLS = mvaQuality;
  otherMVA_t mvaOther_HLS = mvaOther;

  // pack the track word
  setTrackWord(valid_HLS,
               rInv_HLS,
               localPhi_HLS,
               tanl_HLS,
               z0_HLS,
               d0_HLS,
               chi2RPhiRed_HLS,
               chi2RZRed_HLS,
               bendChi2Red_HLS,
               hitPattern_HLS,
               mvaQuality_HLS,
               mvaOther_HLS);
}

// A setting for already-digitized values in HLS format
void TTTrack_TrackWord::setTrackWord(ap_uint<TrackBitWidths::kValidSize> valid,
                                     ap_uint<TrackBitWidths::kRinvSize> rInv,
                                     ap_uint<TrackBitWidths::kPhiSize> localPhi,
                                     ap_uint<TrackBitWidths::kTanlSize> tanl,
                                     ap_uint<TrackBitWidths::kZ0Size> z0,
                                     ap_uint<TrackBitWidths::kD0Size> d0,
                                     ap_uint<TrackBitWidths::kChi2RPhiSize> chi2RPhiRed,
                                     ap_uint<TrackBitWidths::kChi2RZSize> chi2RZRed,
                                     ap_uint<TrackBitWidths::kBendChi2Size> bendChi2Red,
                                     ap_uint<TrackBitWidths::kHitPatternSize> hitPattern,
                                     ap_uint<TrackBitWidths::kMVAQualitySize> mvaQuality,
                                     ap_uint<TrackBitWidths::kMVAOtherSize> mvaOther) {
  // pack the track word
  for (unsigned int b = 0; b < TrackBitWidths::kMVAOtherSize; b++) {
    trackWord_.set(TrackBitLocations::kMVAOtherLSB + b, mvaOther[b]);
  }
  for (unsigned int b = 0; b < TrackBitWidths::kMVAQualitySize; b++) {
    trackWord_.set(TrackBitLocations::kMVAQualityLSB + b, mvaQuality[b]);
  }
  for (unsigned int b = 0; b < TrackBitWidths::kHitPatternSize; b++) {
    trackWord_.set(TrackBitLocations::kHitPatternLSB + b, hitPattern[b]);
  }
  for (unsigned int b = 0; b < TrackBitWidths::kBendChi2Size; b++) {
    trackWord_.set(TrackBitLocations::kBendChi2LSB + b, bendChi2Red[b]);
  }
  for (unsigned int b = 0; b < TrackBitWidths::kD0Size; b++) {
    trackWord_.set(TrackBitLocations::kD0LSB + b, d0[b]);
  }
  for (unsigned int b = 0; b < TrackBitWidths::kChi2RZSize; b++) {
    trackWord_.set(TrackBitLocations::kChi2RZLSB + b, chi2RZRed[b]);
  }
  for (unsigned int b = 0; b < TrackBitWidths::kZ0Size; b++) {
    trackWord_.set(TrackBitLocations::kZ0LSB + b, z0[b]);
  }
  for (unsigned int b = 0; b < TrackBitWidths::kTanlSize; b++) {
    trackWord_.set(TrackBitLocations::kTanlLSB + b, tanl[b]);
  }
  for (unsigned int b = 0; b < TrackBitWidths::kChi2RPhiSize; b++) {
    trackWord_.set(TrackBitLocations::kChi2RPhiLSB + b, chi2RPhiRed[b]);
  }
  for (unsigned int b = 0; b < TrackBitWidths::kPhiSize; b++) {
    trackWord_.set(TrackBitLocations::kPhiLSB + b, localPhi[b]);
  }
  for (unsigned int b = 0; b < TrackBitWidths::kRinvSize; b++) {
    trackWord_.set(TrackBitLocations::kRinvLSB + b, rInv[b]);
  }
  for (unsigned int b = 0; b < TrackBitWidths::kValidSize; b++) {
    trackWord_.set(TrackBitLocations::kValidLSB + b, valid[b]);
  }
}

void TTTrack_TrackWord::printTestDigitizationScheme(const unsigned int nBits,
                                                    const double lsb,
                                                    const double floatingPointValue,
                                                    const unsigned int digitizedSignedValue,
                                                    const double undigitizedSignedValue,
                                                    const unsigned int redigitizedSignedValue) const {
  edm::LogInfo("TTTrack_TrackWord") << "testDigitizationScheme: (nBits, lsb) = (" << nBits << ", " << lsb << ")\n"
                                    << "Floating point value = " << floatingPointValue
                                    << "\tDigitized value = " << digitizedSignedValue
                                    << "\tUn-digitized value = " << undigitizedSignedValue
                                    << "\tRe-digitized value = " << redigitizedSignedValue;
}

bool TTTrack_TrackWord::singleDigitizationSchemeTest(const double floatingPointValue,
                                                     const unsigned int nBits,
                                                     const double lsb) const {
  unsigned int digitizedSignedValue = digitizeSignedValue(floatingPointValue, nBits, lsb);
  double undigitizedSignedValue = undigitizeSignedValue(digitizedSignedValue, nBits, lsb);
  unsigned int redigitizedSignedValue = digitizeSignedValue(undigitizedSignedValue, nBits, lsb);
  this->printTestDigitizationScheme(
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
