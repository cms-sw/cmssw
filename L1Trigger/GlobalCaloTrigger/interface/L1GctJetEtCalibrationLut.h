#ifndef L1GCTJETETCALIBRATIONLUT_H_
#define L1GCTJETETCALIBRATIONLUT_H_

#include <boost/cstdint.hpp> //for uint16_t



class L1GctJetEtCalibrationLut
{
public:
  L1GctJetEtCalibrationLut();
  ~L1GctJetEtCalibrationLut();
  
  /// Converts a 10-bit jet energy to a six-bit rank.
  /*! Eta takes a value from 0-10, corresponding to jet regions running from eta=0 to eta=5 */
  uint16_t convertToSixBitRank(uint16_t jetEnergy, uint16_t eta) const;

  uint16_t convertToTenBitRank(uint16_t jetEnergy, uint16_t eta) const;
  
private:

  static const int JET_ENERGY_BITWIDTH = 10;  //must be 6 or more
  
  
  float m_quadraticCoeff;
//  float m_linearCoeff;
//  float m_constant;
  
  
};


#endif /*L1GCTJETETCALIBRATIONLUT_H_*/
