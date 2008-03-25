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
  enum CorrectionFunctionType { POWER_SERIES_CORRECTION, ORCA_STYLE_CORRECTION,
                                OLD_ORCA_STYLE_CORRECTION, NO_CORRECTION };

  static const unsigned NUMBER_ETA_VALUES;     ///< Number of eta bins used in correction
  static const unsigned N_CENTRAL_ETA_VALUES;  ///< Number of eta bins for separate tau correction

  L1GctJetEtCalibrationFunction();
  ~L1GctJetEtCalibrationFunction();

  /// set other parameters
  void setParams(const double& htScale, const double& threshold,
                 const std::vector< std::vector<double> >& jetCalibFunc,
                 const std::vector< std::vector<double> >& tauCalibFunc );

  /// set the type of correction function to use
  void setCorrectionFunctionType(const CorrectionFunctionType cft) { m_corrFunType = cft;
                                                                   }
                                                                   
  /// Overload << operator
  friend std::ostream& operator << (std::ostream& os, const L1GctJetEtCalibrationFunction& fn);
  
  /// apply jet Et correction
  double correctedEt(double et, unsigned eta, bool tauVeto) const;
  
  /// Convert the corrected Et value to a linear Et for Ht summing
  uint16_t calibratedEt(const double Et) const;

  /// Access method for Ht scale
  /// (LSB for integer->physical conversion, in GeV units)
  double getHtScaleLSB() const { return m_htScaleLSB; }

  /// Access method for jet threshold
  /// (in GeV units)
  double getThreshold() const { return m_threshold; }


private:

  // CONVERSION FUNCTIONS
  /// Find the corrected Et value for this jet
  /*! Eta takes a value from 0-10, corresponding to jet regions running from eta=0.0 to eta=5.0 */
  double findCorrectedEt   (const double Et, const std::vector<double>& coeffs) const;
  double powerSeriesCorrect(const double Et, const std::vector<double>& coeffs) const;
  double orcaStyleCorrect  (const double Et, const std::vector<double>& coeffs) const;
  
  /// Convert the corrected Et value to a non-linear jet rank for sorting
  uint16_t rank(const double Et) const;

  float orcaCalibFn(float et, unsigned eta) const;

  // PARAMETERS FOR THE CONVERSION
  /// type of correction function to apply
  CorrectionFunctionType m_corrFunType; 

  /// scale and threshold parameters
  double m_htScaleLSB;
  double m_threshold;

  /// the calibration function - converts jet Et to linear 
  std::vector< std::vector<double> > m_jetCalibFunc;
  std::vector< std::vector<double> > m_tauCalibFunc;

};

std::ostream& operator << (std::ostream& os, const L1GctJetEtCalibrationFunction& fn);

#endif /*L1GCTJETETCALIBRATIONFUNCTION_H_*/
