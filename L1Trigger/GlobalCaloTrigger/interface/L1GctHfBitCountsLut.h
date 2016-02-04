#ifndef L1GCTHFBITCOUNTSLUT_H_
#define L1GCTHFBITCOUNTSLUT_H_

#include "L1Trigger/GlobalCaloTrigger/src/L1GctLut.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctHfEtSumsLut.h"

#include <vector>

/*!
 * \author Greg Heath
 * \date September 2008
 */

/*! \class L1GctHfBitCountsLut
 * \brief LUT for compression of HF feature bit counts to output format
 *
 */


class L1GctHfBitCountsLut : public L1GctLut<5,3>

{
public:

  // Definitions.
  static const int NAddress, NData;

  /// Constructor for use with emulator - which type of Lut?
  L1GctHfBitCountsLut(const L1GctHfEtSumsLut::hfLutType& type);
  /// Default constructor
  L1GctHfBitCountsLut();
  /// Copy constructor
  L1GctHfBitCountsLut(const L1GctHfBitCountsLut& lut);
  /// Destructor
  virtual ~L1GctHfBitCountsLut();
  
  /// Overload = operator
  L1GctHfBitCountsLut operator= (const L1GctHfBitCountsLut& lut);

  /// Overload << operator
  friend std::ostream& operator << (std::ostream& os, const L1GctHfBitCountsLut& lut);

  /// Return the type of Lut
  L1GctHfEtSumsLut::hfLutType lutType() const { return m_lutType; }

  /// Get thresholds
  std::vector<unsigned> getThresholdsGct() const;

protected:
  

  virtual uint16_t value (const uint16_t lutAddress) const;

private:

  L1GctHfEtSumsLut::hfLutType m_lutType;
  
};


std::ostream& operator << (std::ostream& os, const L1GctHfBitCountsLut& lut);

#endif /*L1GCTHFBITCOUNTSLUT_H_*/
