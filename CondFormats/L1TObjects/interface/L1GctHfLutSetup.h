#ifndef L1GCTHFLUTSETUP_H_
#define L1GCTHFLUTSETUP_H_

/*!
 * \author Greg Heath
 * \date Sep 2007
 */

#include <vector>
#include <map>
#include <stdint.h>
#include <iosfwd>

/*! \class L1GctHfLutSetup
 * \brief Hf Lut setup (all the Luts in one class)
 * 
 *
 *============================================================================
 *
 *
 *============================================================================
 *
 */


class L1GctHfLutSetup
{
public:

  /// Define the different types of Lut
  enum hfLutType { bitCountPosEtaRing1, bitCountPosEtaRing2, bitCountNegEtaRing1, bitCountNegEtaRing2,
                      etSumPosEtaRing1,    etSumPosEtaRing2,    etSumNegEtaRing1,    etSumNegEtaRing2,
                                                                                     numberOfLutTypes};
  /// Define the numbers of input and output bits
  enum numberOfEtSumBits {
    kHfEtSumBits      = 8,
    kHfEtSumNumValues = 1 << kHfEtSumBits,
    kHfEtSumMaxValue  = kHfEtSumNumValues - 1
  };

  enum numberOfCountBits {
    kHfCountBits      = 5,
    kHfCountNumValues = 1 << kHfCountBits,
    kHfCountMaxValue  = kHfCountNumValues - 1
  };

  enum numberOfOutputBits {
    kHfOutputBits      = 3,
    kHfOutputNumValues = 1 << kHfOutputBits,
    kHfOutputMaxValue  = kHfOutputNumValues - 1
  };


  L1GctHfLutSetup();
  ~L1GctHfLutSetup();

  // Set method for the thresholds
  void setThresholds(const hfLutType type, const std::vector<unsigned> thr);

  // Accessor methods
  std::vector<unsigned> getThresholds(const hfLutType type) const;

  std::vector<unsigned> getCountThresholdsPosEtaRing1() const { return getThresholds(bitCountPosEtaRing1); }
  std::vector<unsigned> getCountThresholdsPosEtaRing2() const { return getThresholds(bitCountPosEtaRing2); }
  std::vector<unsigned> getCountThresholdsNegEtaRing1() const { return getThresholds(bitCountNegEtaRing1); }
  std::vector<unsigned> getCountThresholdsNegEtaRing2() const { return getThresholds(bitCountNegEtaRing2); }
  std::vector<unsigned> getEtSumThresholdsPosEtaRing1() const { return getThresholds(etSumPosEtaRing1); }
  std::vector<unsigned> getEtSumThresholdsPosEtaRing2() const { return getThresholds(etSumPosEtaRing2); }
  std::vector<unsigned> getEtSumThresholdsNegEtaRing1() const { return getThresholds(etSumNegEtaRing1); }
  std::vector<unsigned> getEtSumThresholdsNegEtaRing2() const { return getThresholds(etSumNegEtaRing2); }

  // This is the LUT result
  uint16_t outputValue(const hfLutType type, const uint16_t inputValue) const;

private:

  std::map<hfLutType, std::vector<uint16_t> > m_thresholds;

};

std::ostream& operator << (std::ostream& os, const L1GctHfLutSetup& fn);

#endif /*L1GCTHFLUTSETUP_H_*/
