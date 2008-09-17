#ifndef L1GCTHFETSUMSLUT_H_
#define L1GCTHFETSUMSLUT_H_

#include "CondFormats/L1TObjects/interface/L1GctHfLutSetup.h"

#include "L1Trigger/GlobalCaloTrigger/src/L1GctLut.h"

#include <vector>

/*!
 * \author Greg Heath
 * \date September 2008
 */

/*! \class L1GctHfEtSumsLut
 * \brief LUT for compression of HF Et sum to output format
 *
 */


class L1GctHfEtSumsLut : public L1GctLut<L1GctHfLutSetup::kHfEtSumBits,
                                         L1GctHfLutSetup::kHfOutputBits>

{
public:

  // Definitions.
  static const int NAddress, NData;

  /// Constructor for use with emulator - which type of Lut?
  L1GctHfEtSumsLut(const L1GctHfLutSetup::hfLutType& type, const L1GctHfLutSetup* const fn);
  /// Constructor for use with emulator - which type of Lut?
  L1GctHfEtSumsLut(const L1GctHfLutSetup::hfLutType& type);
  /// Default constructor
  L1GctHfEtSumsLut();
  /// Copy constructor
  L1GctHfEtSumsLut(const L1GctHfEtSumsLut& lut);
  /// Destructor
  virtual ~L1GctHfEtSumsLut();
  
  /// Overload = operator
  L1GctHfEtSumsLut operator= (const L1GctHfEtSumsLut& lut);

  /// Overload << operator
  friend std::ostream& operator << (std::ostream& os, const L1GctHfEtSumsLut& lut);

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


std::ostream& operator << (std::ostream& os, const L1GctHfEtSumsLut& lut);

#endif /*L1GCTHFETSUMSLUT_H_*/
