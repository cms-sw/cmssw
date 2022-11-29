#ifndef L1_TRACK_TRIGGER_TRACK_WORD_H
#define L1_TRACK_TRIGGER_TRACK_WORD_H

////////
//
// class to store the 96-bit track word produced by the L1 Track Trigger.  Intended to be inherited by L1 TTTrack.
// packing scheme given below.
//
// author:      Mike Hildreth
// modified by: Alexx Perloff
// created:     April 9, 2019
// modified:    March 9, 2021
//
///////

#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include <ap_int.h>

#include <algorithm>
#include <array>
#include <bitset>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace tttrack_trackword {
  void infoTestDigitizationScheme(
      const unsigned int, const double, const double, const unsigned int, const double, const unsigned int);
}

class TTTrack_TrackWord {
public:
  // ----------constants, enums and typedefs ---------
  enum TrackBitWidths {
    // The sizes of the track word components
    kMVAOtherSize = 6,    // Space for two specialized MVA selections
    kMVAQualitySize = 3,  // Width of track quality MVA
    kHitPatternSize = 7,  // Width of the hit pattern for stubs
    kBendChi2Size = 3,    // Width of the bend-chi2/dof
    kD0Size = 13,         // Width of D0
    kChi2RZSize = 4,      // Width of chi2/dof for r-z
    kZ0Size = 12,         // Width of z-position (40cm / 0.1)
    kTanlSize = 16,       // Width of tan(lambda)
    kChi2RPhiSize = 4,    // Width of chi2/dof for r-phi
    kPhiSize = 12,        // Width of phi
    kRinvSize = 15,       // Width of Rinv
    kValidSize = 1,       // Valid bit

    kTrackWordSize = kValidSize + kRinvSize + kPhiSize + kChi2RPhiSize + kTanlSize + kZ0Size + kChi2RZSize + kD0Size +
                     kBendChi2Size + kHitPatternSize + kMVAQualitySize +
                     kMVAOtherSize,  // Width of the track word in bits
  };

  enum TrackBitLocations {
    // The location of the least significant bit (LSB) and most significant bit (MSB) in the track word for different fields
    kMVAOtherLSB = 0,
    kMVAOtherMSB = kMVAOtherLSB + TrackBitWidths::kMVAOtherSize - 1,
    kMVAQualityLSB = kMVAOtherMSB + 1,
    kMVAQualityMSB = kMVAQualityLSB + TrackBitWidths::kMVAQualitySize - 1,
    kHitPatternLSB = kMVAQualityMSB + 1,
    kHitPatternMSB = kHitPatternLSB + TrackBitWidths::kHitPatternSize - 1,
    kBendChi2LSB = kHitPatternMSB + 1,
    kBendChi2MSB = kBendChi2LSB + TrackBitWidths::kBendChi2Size - 1,
    kD0LSB = kBendChi2MSB + 1,
    kD0MSB = kD0LSB + TrackBitWidths::kD0Size - 1,
    kChi2RZLSB = kD0MSB + 1,
    kChi2RZMSB = kChi2RZLSB + TrackBitWidths::kChi2RZSize - 1,
    kZ0LSB = kChi2RZMSB + 1,
    kZ0MSB = kZ0LSB + TrackBitWidths::kZ0Size - 1,
    kTanlLSB = kZ0MSB + 1,
    kTanlMSB = kTanlLSB + TrackBitWidths::kTanlSize - 1,
    kChi2RPhiLSB = kTanlMSB + 1,
    kChi2RPhiMSB = kChi2RPhiLSB + TrackBitWidths::kChi2RPhiSize - 1,
    kPhiLSB = kChi2RPhiMSB + 1,
    kPhiMSB = kPhiLSB + TrackBitWidths::kPhiSize - 1,
    kRinvLSB = kPhiMSB + 1,
    kRinvMSB = kRinvLSB + TrackBitWidths::kRinvSize - 1,
    kValidLSB = kRinvMSB + 1,
    kValidMSB = kValidLSB + TrackBitWidths::kValidSize - 1,
  };

  // Binning constants
  static constexpr double minRinv = -0.006;
  static constexpr double minPhi0 = -0.7853981696;  // relative to the center of the sector
  static constexpr double minTanl = -8.;
  static constexpr double minZ0 = -20.46912512;
  static constexpr double minD0 = -16.;

  static constexpr double stepRinv = (2. * std::abs(minRinv)) / (1 << TrackBitWidths::kRinvSize);
  static constexpr double stepPhi0 = (2. * std::abs(minPhi0)) / (1 << TrackBitWidths::kPhiSize);
  static constexpr double stepTanL = (1. / (1 << 12));
  static constexpr double stepZ0 = (2. * std::abs(minZ0)) / (1 << TrackBitWidths::kZ0Size);
  static constexpr double stepD0 = (1. / (1 << 8));

  // Bin edges for chi2/dof
  static constexpr std::array<double, 1 << TrackBitWidths::kChi2RPhiSize> chi2RPhiBins = {
      {0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 10.0, 15.0, 20.0, 35.0, 60.0, 200.0}};
  static constexpr std::array<double, 1 << TrackBitWidths::kChi2RZSize> chi2RZBins = {
      {0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 8.0, 10.0, 20.0, 50.0}};
  static constexpr std::array<double, 1 << TrackBitWidths::kBendChi2Size> bendChi2Bins = {
      {0.0, 0.75, 1.0, 1.5, 2.25, 3.5, 5.0, 20.0}};

  // Sector constants
  static constexpr unsigned int nSectors = 9;
  static constexpr double sectorWidth = (2. * M_PI) / nSectors;

  // Track flags
  typedef ap_uint<TrackBitWidths::kValidSize> valid_t;  // valid bit

  // Track parameters types
  typedef ap_uint<TrackBitWidths::kRinvSize> rinv_t;  // Track Rinv
  typedef ap_uint<TrackBitWidths::kPhiSize> phi_t;    // Track phi
  typedef ap_uint<TrackBitWidths::kTanlSize> tanl_t;  // Track tan(l)
  typedef ap_uint<TrackBitWidths::kZ0Size> z0_t;      // Track z
  typedef ap_uint<TrackBitWidths::kD0Size> d0_t;      // D0

  // Track quality types
  typedef ap_uint<TrackBitWidths::kChi2RPhiSize> chi2rphi_t;      // Chi2 r-phi
  typedef ap_uint<TrackBitWidths::kChi2RZSize> chi2rz_t;          // Chi2 r-z
  typedef ap_uint<TrackBitWidths::kBendChi2Size> bendChi2_t;      // Bend-Chi2
  typedef ap_uint<TrackBitWidths::kHitPatternSize> hit_t;         // Hit mask bits
  typedef ap_uint<TrackBitWidths::kMVAQualitySize> qualityMVA_t;  // Track quality MVA
  typedef ap_uint<TrackBitWidths::kMVAOtherSize> otherMVA_t;      // Specialized MVA selection

  // Track word types
  typedef std::bitset<TrackBitWidths::kTrackWordSize> tkword_bs_t;  // Entire track word;
  typedef ap_uint<TrackBitWidths::kTrackWordSize> tkword_t;         // Entire track word;

public:
  // ----------Constructors --------------------------
  TTTrack_TrackWord() {}
  TTTrack_TrackWord(unsigned int valid,
                    const GlobalVector& momentum,
                    const GlobalPoint& POCA,
                    double rInv,
                    double chi2RPhi,  // would be xy chisq if chi2Z is non-zero
                    double chi2RZ,
                    double bendChi2,
                    unsigned int hitPattern,
                    unsigned int mvaQuality,
                    unsigned int mvaOther,
                    unsigned int sector);
  TTTrack_TrackWord(unsigned int valid,
                    unsigned int rInv,
                    unsigned int phi0,  // local phi
                    unsigned int tanl,
                    unsigned int z0,
                    unsigned int d0,
                    unsigned int chi2RPhi,  // would be total chisq if chi2Z is zero
                    unsigned int chi2RZ,
                    unsigned int bendChi2,
                    unsigned int hitPattern,
                    unsigned int mvaQuality,
                    unsigned int mvaOther);

  // ----------copy constructor ----------------------
  TTTrack_TrackWord(const TTTrack_TrackWord& word) { trackWord_ = word.trackWord_; }

  // ----------operators -----------------------------
  TTTrack_TrackWord& operator=(const TTTrack_TrackWord& word) {
    trackWord_ = word.trackWord_;
    return *this;
  }

  // ----------member functions (getters) ------------
  // These functions return arbitarary precision unsigned int words (lists of bits) for each quantity
  // Signed quantities have the sign enconded in the left-most bit.
  valid_t getValidWord() const { return getTrackWord()(TrackBitLocations::kValidMSB, TrackBitLocations::kValidLSB); }
  rinv_t getRinvWord() const { return getTrackWord()(TrackBitLocations::kRinvMSB, TrackBitLocations::kRinvLSB); }
  phi_t getPhiWord() const { return getTrackWord()(TrackBitLocations::kPhiMSB, TrackBitLocations::kPhiLSB); }
  tanl_t getTanlWord() const { return getTrackWord()(TrackBitLocations::kTanlMSB, TrackBitLocations::kTanlLSB); }
  z0_t getZ0Word() const { return getTrackWord()(TrackBitLocations::kZ0MSB, TrackBitLocations::kZ0LSB); }
  d0_t getD0Word() const { return getTrackWord()(TrackBitLocations::kD0MSB, TrackBitLocations::kD0LSB); }
  chi2rphi_t getChi2RPhiWord() const {
    return getTrackWord()(TrackBitLocations::kChi2RPhiMSB, TrackBitLocations::kChi2RPhiLSB);
  }
  chi2rz_t getChi2RZWord() const {
    return getTrackWord()(TrackBitLocations::kChi2RZMSB, TrackBitLocations::kChi2RZLSB);
  }
  bendChi2_t getBendChi2Word() const {
    return getTrackWord()(TrackBitLocations::kBendChi2MSB, TrackBitLocations::kBendChi2LSB);
  }
  hit_t getHitPatternWord() const {
    return getTrackWord()(TrackBitLocations::kHitPatternMSB, TrackBitLocations::kHitPatternLSB);
  }
  qualityMVA_t getMVAQualityWord() const {
    return getTrackWord()(TrackBitLocations::kMVAQualityMSB, TrackBitLocations::kMVAQualityLSB);
  }
  otherMVA_t getMVAOtherWord() const {
    return getTrackWord()(TrackBitLocations::kMVAOtherMSB, TrackBitLocations::kMVAOtherLSB);
  }
  tkword_t getTrackWord() const { return tkword_t(trackWord_.to_string().c_str(), 2); }

  // These functions return the packed bits in unsigned integer format for each quantity
  // Signed quantities have the sign enconded in the left-most bit of the pattern using
  //   a two's complement representation
  unsigned int getValidBits() const { return getValidWord().to_uint(); }
  unsigned int getRinvBits() const { return getRinvWord().to_uint(); }
  unsigned int getPhiBits() const { return getPhiWord().to_uint(); }
  unsigned int getTanlBits() const { return getTanlWord().to_uint(); }
  unsigned int getZ0Bits() const { return getZ0Word().to_uint(); }
  unsigned int getD0Bits() const { return getD0Word().to_uint(); }
  unsigned int getChi2RPhiBits() const { return getChi2RPhiWord().to_uint(); }
  unsigned int getChi2RZBits() const { return getChi2RZWord().to_uint(); }
  unsigned int getBendChi2Bits() const { return getBendChi2Word().to_uint(); }
  unsigned int getHitPatternBits() const { return getHitPatternWord().to_uint(); }
  unsigned int getMVAQualityBits() const { return getMVAQualityWord().to_uint(); }
  unsigned int getMVAOtherBits() const { return getMVAOtherWord().to_uint(); }

  // These functions return the unpacked and converted values
  // These functions return real numbers converted from the digitized quantities by unpacking the 96-bit track word
  bool getValid() const { return getValidWord().to_bool(); }
  double getRinv() const { return undigitizeSignedValue(getRinvBits(), TrackBitWidths::kRinvSize, stepRinv); }
  double getPhi() const { return undigitizeSignedValue(getPhiBits(), TrackBitWidths::kPhiSize, stepPhi0); }
  double getTanl() const { return undigitizeSignedValue(getTanlBits(), TrackBitWidths::kTanlSize, stepTanL); }
  double getZ0() const { return undigitizeSignedValue(getZ0Bits(), TrackBitWidths::kZ0Size, stepZ0); }
  double getD0() const { return undigitizeSignedValue(getD0Bits(), TrackBitWidths::kD0Size, stepD0); }
  double getChi2RPhi() const { return chi2RPhiBins[getChi2RPhiBits()]; }
  double getChi2RZ() const { return chi2RZBins[getChi2RZBits()]; }
  double getBendChi2() const { return bendChi2Bins[getBendChi2Bits()]; }
  unsigned int getHitPattern() const { return getHitPatternBits(); }
  unsigned int getNStubs() const { return countSetBits(getHitPatternBits()); }
  unsigned int getMVAQuality() const { return getMVAQualityBits(); }
  unsigned int getMVAOther() const { return getMVAOtherBits(); }

  // ----------member functions (setters) ------------
  void setTrackWord(unsigned int valid,
                    const GlobalVector& momentum,
                    const GlobalPoint& POCA,
                    double rInv,
                    double chi2RPhi,  // would be total chisq if chi2Z is zero
                    double chi2RZ,
                    double bendChi2,
                    unsigned int hitPattern,
                    unsigned int mvaQuality,
                    unsigned int mvaOther,
                    unsigned int sector);

  void setTrackWord(unsigned int valid,
                    unsigned int rInv,
                    unsigned int phi0,  // local phi
                    unsigned int tanl,
                    unsigned int z0,
                    unsigned int d0,
                    unsigned int chi2RPhi,  // would be total chisq if chi2Z is zero
                    unsigned int chi2RZ,
                    unsigned int bendChi2,
                    unsigned int hitPattern,
                    unsigned int mvaQuality,
                    unsigned int mvaOther);

  void setTrackWord(ap_uint<TrackBitWidths::kValidSize> valid,
                    ap_uint<TrackBitWidths::kRinvSize> rInv,
                    ap_uint<TrackBitWidths::kPhiSize> phi0,  // local phi
                    ap_uint<TrackBitWidths::kTanlSize> tanl,
                    ap_uint<TrackBitWidths::kZ0Size> z0,
                    ap_uint<TrackBitWidths::kD0Size> d0,
                    ap_uint<TrackBitWidths::kChi2RPhiSize> chi2RPhi,  // would be total chisq if chi2Z is zero
                    ap_uint<TrackBitWidths::kChi2RZSize> chi2RZ,
                    ap_uint<TrackBitWidths::kBendChi2Size> bendChi2,
                    ap_uint<TrackBitWidths::kHitPatternSize> hitPattern,
                    ap_uint<TrackBitWidths::kMVAQualitySize> mvaQuality,
                    ap_uint<TrackBitWidths::kMVAOtherSize> mvaOther);

  // ----------member functions (testers) ------------
  bool singleDigitizationSchemeTest(const double floatingPointValue, const unsigned int nBits, const double lsb) const;
  void testDigitizationScheme() const;

protected:
  // ----------protected member functions ------------
  float localPhi(float globalPhi, unsigned int sector) const {
    return reco::deltaPhi(globalPhi, (sector * sectorWidth));
  }

public:
  // ----------public member functions --------------
  unsigned int countSetBits(unsigned int n) const {
    // Adapted from: https://www.geeksforgeeks.org/count-set-bits-in-an-integer/
    unsigned int count = 0;
    while (n) {
      n &= (n - 1);
      count++;
    }
    return count;
  }

  unsigned int digitizeSignedValue(double value, unsigned int nBits, double lsb) const {
    // Digitize the incoming value
    int digitizedValue = std::floor(value / lsb);

    // Calculate the maxmum possible positive value given an output of nBits in size
    int digitizedMaximum = (1 << (nBits - 1)) - 1;  // The remove 1 bit from nBits to account for the sign
    int digitizedMinimum = -1. * (digitizedMaximum + 1);

    // Saturate the digitized value
    digitizedValue = std::clamp(digitizedValue, digitizedMinimum, digitizedMaximum);

    // Do the two's compliment encoding
    unsigned int twosValue = digitizedValue;
    if (digitizedValue < 0) {
      twosValue += (1 << nBits);
    }

    return twosValue;
  }

  template <typename T>
  constexpr unsigned int getBin(double value, const T& bins) const {
    auto up = std::upper_bound(bins.begin(), bins.end(), value);
    return (up - bins.begin() - 1);
  }

  double undigitizeSignedValue(unsigned int twosValue, unsigned int nBits, double lsb, double offset = 0.5) const {
    // Check that none of the bits above the nBits-1 bit, in a range of [0, nBits-1], are set.
    // This makes sure that it isn't possible for the value represented by `twosValue` to be
    //  any bigger than ((1 << nBits) - 1).
    assert((twosValue >> nBits) == 0);

    // Convert from twos compliment to C++ signed integer (normal digitized value)
    int digitizedValue = twosValue;
    if (twosValue & (1 << (nBits - 1))) {  // check if the twosValue is negative
      digitizedValue -= (1 << nBits);
    }

    // Convert to floating point value
    return (double(digitizedValue) + offset) * lsb;
  }

  // ----------member data ---------------------------
  tkword_bs_t trackWord_;
};

#endif
