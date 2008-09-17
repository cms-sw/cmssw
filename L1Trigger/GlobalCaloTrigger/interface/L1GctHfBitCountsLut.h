#ifndef L1GCTHFBITCOUNTSLUT_H_
#define L1GCTHFBITCOUNTSLUT_H_

#include "CondFormats/L1TObjects/interface/L1GctHfLutSetup.h"

#include "L1Trigger/GlobalCaloTrigger/src/L1GctLut.h"

#include <vector>

/*!
 * \author Greg Heath
 * \date September 2008
 */

/*! \class L1GctHfBitCountsLut
 * \brief LUT for compression of HF feature bit counts to output format
 *
 */


class L1GctHfBitCountsLut : public L1GctLut<L1GctHfLutSetup::kHfCountBits,
                                            L1GctHfLutSetup::kHfOutputBits>

{
public:

  // Definitions.
  static const int NAddress, NData;

  /// Constructor for use with emulator - which type of Lut?
  L1GctHfBitCountsLut(const L1GctHfLutSetup::hfLutType& type, const L1GctHfLutSetup* const fn);
  /// Constructor for use with emulator - which type of Lut?
  L1GctHfBitCountsLut(const L1GctHfLutSetup::hfLutType& type);
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

  /// Set the function
  void setFunction(const L1GctHfLutSetup* const fn) { if (fn != 0) { m_lutFunction = fn; m_setupOk = true; } }

  /// Return the type of Lut
  L1GctHfLutSetup::hfLutType lutType() const { return m_lutType; }

  /// Return the Lut function
  const L1GctHfLutSetup* lutFunction() const { return m_lutFunction; }

protected:
  

  virtual uint16_t value (const uint16_t lutAddress) const;

private:

  const L1GctHfLutSetup* m_lutFunction;
  L1GctHfLutSetup::hfLutType m_lutType;
  
};


std::ostream& operator << (std::ostream& os, const L1GctHfBitCountsLut& lut);

#endif /*L1GCTHFBITCOUNTSLUT_H_*/
