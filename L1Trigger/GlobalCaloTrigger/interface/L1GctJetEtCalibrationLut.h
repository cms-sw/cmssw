#ifndef L1GCTJETETCALIBRATIONLUT_H_
#define L1GCTJETETCALIBRATIONLUT_H_

#include "FWCore/Framework/interface/ESHandle.h"

#include <boost/cstdint.hpp> //for uint16_t

#include <vector>
#include <string>

class L1CaloEtScale;

/*!
 * \author Robert Frazier
 * \date May 2006
 */

/*! \class L1GctJetEtCalibrationLut
 * \brief Jet Et calibration LUT
 * 
 * Input is 10 bit Et and 4 bit eta
 * Outputs are 6 bit rank (for jet sorting) and 10 bit Et (for Ht calculation)
 * 
 * Currently just performs a truncation from 10 to 6 bits, no eta dependence
 *
 */


class L1GctJetEtCalibrationLut
{
public:
  static const unsigned JET_ENERGY_BITWIDTH;  ///< Input bitwidth of jet energy; must be 10 or more
  static const unsigned NUMBER_ETA_VALUES;  ///< Number of eta bins used in correction
  
  L1GctJetEtCalibrationLut();
  L1GctJetEtCalibrationLut(std::string fileName);
  ~L1GctJetEtCalibrationLut();

  /// set the output Et scale pointer
  void setOutputEtScale(const L1CaloEtScale* scale);

  /// Overload << operator
  friend std::ostream& operator << (std::ostream& os, const L1GctJetEtCalibrationLut& lut);
  
  /// Converts a 10-bit jet Et to a 6-bit rank.
  /*! Eta takes a value from 0-10, corresponding to jet regions running from eta=0 to eta=5 */
  uint16_t rank(const uint16_t jetEt, const unsigned eta) const;

  /// Converts a 10-bit jet Et to a 10-bit Et (applying eta-dependent calibration)
  /*! Eta takes a value from 0-10, corresponding to jet regions running from eta=0 to eta=5 */
  uint16_t calibratedEt(const uint16_t jetEt, const unsigned eta) const;
  
private:

  /// the output scale - converts linear Et to rank
  const L1CaloEtScale* m_outputEtScale;

  /// the calibration function
  std::vector< std::vector<float> > m_calibFunc;

};

std::ostream& operator << (std::ostream& os, const L1GctJetEtCalibrationLut& lut);

#endif /*L1GCTJETETCALIBRATIONLUT_H_*/
