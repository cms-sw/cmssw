#ifndef L1GCTJETETCALIBRATIONFUNCTION_H_
#define L1GCTJETETCALIBRATIONFUNCTION_H_

#include <boost/cstdint.hpp> //for uint16_t

#include <vector>
#include <string>

/*!
 * \author Robert Frazier and Greg Heath
 * \date Mar 2007
 */

/*! \class L1GctJetEtCalibrationFunction
 * \brief Jet Et calibration Function
 * 
 * Input is measured Et (in GeV), 4 bit eta, and tau veto flag 
 * Output is corrected Et (in GeV) and an integer version for Ht summing
 * 
 * This used to be part of JetEtCalibrationLut, now separated out on its own.
 *
 *============================================================================
 *
 * The external entry points are
 *
 *    double   L1GctJetCalibrationFunction::correctedEt (const double et, const unsigned eta, const bool tauVeto) const;
 *    uint16_t L1GctJetCalibrationFunction::calibratedEt(const double correctedEt) const;
 *
 *============================================================================
 *
 */


class L1GctJetEtCalibrationFunction
{
public:
  enum CorrectionFunctionType { POWER_SERIES_CORRECTION,
                                ORCA_STYLE_CORRECTION,
                                PIECEWISE_CUBIC_CORRECTION,
                                NO_CORRECTION };

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
  /// Eta takes a value from 0-10, corresponding to jet regions running from eta=0.0 to eta=5.0
  double correctedEt(const double et, const unsigned eta, const bool tauVeto) const;
  
  /// Convert the corrected Et value to a linear Et for Ht summing
  uint16_t calibratedEt(const double correctedEt) const;

  /// Access method for Ht scale
  /// (LSB for integer->physical conversion, in GeV units)
  double getHtScaleLSB() const { return m_htScaleLSB; }

  /// Access method for jet threshold
  /// (in GeV units)
  double getThreshold() const { return m_threshold; }


private:

  // CONVERSION FUNCTIONS
  /// Find the corrected Et value for this jet
  double findCorrectedEt       (const double Et, const std::vector<double>& coeffs) const;
  double powerSeriesCorrect    (const double Et, const std::vector<double>& coeffs) const;
  double orcaStyleCorrect      (const double Et, const std::vector<double>& coeffs) const;
  double piecewiseCubicCorrect (const double Et, const std::vector<double>& coeffs) const;
  
  /// Convert the corrected Et value to a non-linear jet rank for sorting
  uint16_t rank(const double Et) const;

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
