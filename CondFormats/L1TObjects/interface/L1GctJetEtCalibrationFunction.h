#ifndef L1GCTJETETCALIBRATIONFUNCTION_H_
#define L1GCTJETETCALIBRATIONFUNCTION_H_

#include <boost/cstdint.hpp> //for uint16_t

#include <vector>
#include <string>

#include "CondFormats/L1TObjects/interface/L1CaloEtScale.h"

/*!
 * \author Robert Frazier and Greg Heath
 * \date Mar 2007
 */

/*! \class L1GctJetEtCalibrationFunction
 * \brief Jet Et calibration Function
 * 
 * Input is 10 bit Et, 4 bit eta, and tau veto flag 
 * Outputs are 6 bit rank (for jet sorting) and 10 bit Et (for Ht calculation)
 * 
 * This used to be part of JetEtCalibrationLut, now separated out on its own.
 *
 *============================================================================
 *
 * Algorithm description - the calculation has the following steps:
 *
 * 1) Both input and output are packed into 16-bit fields. Start by unpacking
 *    the jet Et, eta and tau flag from the input Lut address.
 * 2) Multiply by the RCT output LSB to convert Et to a real value.
 * 3) Apply corrections to Et. These depend on both eta and the tau flag.
 * 4) Check whether Et is above the zero suppression threshold. If not, return
 *    a value of zero.
 * 5) Separately calculate the jet rank and the Ht contribution. Rank is 6-bits,
 *    assigned on a non-linear scale determined by the L1JetEtScaleRcd.
 *    Ht is 10-bits linear, with the Lsb stored in this class.
 * 6) Pack the two values into the 16-bit output field, with Ht in bits 9:0,
 *    and rank in bits 15:10. Return the result.
 *
 * The external entry point is
 *
 *    uint16_t L1GctJetCalibrationFunction::lutValue(const uint16_t lutAddress) const;
 *
 *============================================================================
 *
 */


class L1GctJetEtCalibrationFunction
{
public:
  static const unsigned JET_ENERGY_BITWIDTH;   ///< Input bitwidth of jet energy; must be 10 or more
  static const unsigned NUMBER_ETA_VALUES;     ///< Number of eta bins used in correction
  static const unsigned N_CENTRAL_ETA_VALUES;  ///< Number of eta bins for separate tau correction
  
  L1GctJetEtCalibrationFunction();
  ~L1GctJetEtCalibrationFunction();

  /// set the output Et scale pointer
  void setOutputEtScale(const L1CaloEtScale& scale);

  /// set other parameters
  void setParams(const double& htScale, const double& threshold,
                 const std::vector< std::vector<double> >& jetCalibFunc,
                 const std::vector< std::vector<double> >& tauCalibFunc );

  /// Overload << operator
  friend std::ostream& operator << (std::ostream& os, const L1GctJetEtCalibrationFunction& fn);
  
  /// Here's the public interface to the lut values
  uint16_t lutValue(const uint16_t lutAddress) const;
  
private:

  // CONVERSION FUNCTIONS
  /// Find the corrected Et value for this jet
  /*! Eta takes a value from 0-10, corresponding to jet regions running from eta=0.0 to eta=5.0 */
  double correctedEt(const uint16_t jetEt, const unsigned eta, const bool tauVeto) const;
  double powerSeriesCorrect(const double Et, const std::vector<double> coeffs) const;

  /// Convert the corrected Et value to a non-linear jet rank for sorting
  uint16_t rank(const double Et) const;

  /// Convert the corrected Et value to a linear Et for Ht summing
  uint16_t calibratedEt(const double Et) const;

  // PARAMETERS FOR THE CONVERSION
  /// the output scale - converts linear Et to rank
  L1CaloEtScale m_outputEtScale;

  /// scale and threshold parameters
  double m_htScaleLSB;
  double m_threshold;

  /// the calibration function - converts jet Et to linear 
  std::vector< std::vector<double> > m_jetCalibFunc;
  std::vector< std::vector<double> > m_tauCalibFunc;

};

std::ostream& operator << (std::ostream& os, const L1GctJetEtCalibrationFunction& fn);

#endif /*L1GCTJETETCALIBRATIONFUNCTION_H_*/
