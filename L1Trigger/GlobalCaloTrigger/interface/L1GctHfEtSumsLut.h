#ifndef L1GCTHFETSUMSLUT_H_
#define L1GCTHFETSUMSLUT_H_

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

class L1CaloEtScale;

class L1GctHfEtSumsLut : public L1GctLut<8, 3>

{
public:
  enum hfLutType {
    bitCountPosEtaRing1,
    bitCountPosEtaRing2,
    bitCountNegEtaRing1,
    bitCountNegEtaRing2,
    etSumPosEtaRing1,
    etSumPosEtaRing2,
    etSumNegEtaRing1,
    etSumNegEtaRing2,
    numberOfLutTypes
  };

  // Definitions.
  static const int NAddress, NData;

  /// Constructor for use with emulator - which type of Lut?
  L1GctHfEtSumsLut(const L1GctHfEtSumsLut::hfLutType& type, const L1CaloEtScale* const scale);
  /// Constructor for use with emulator - which type of Lut?
  L1GctHfEtSumsLut(const L1GctHfEtSumsLut::hfLutType& type);
  /// Default constructor
  L1GctHfEtSumsLut();
  /// Copy constructor
  L1GctHfEtSumsLut(const L1GctHfEtSumsLut& lut);
  /// Destructor
  ~L1GctHfEtSumsLut() override;

  /// Overload = operator
  L1GctHfEtSumsLut operator=(const L1GctHfEtSumsLut& lut);

  /// Overload << operator
  friend std::ostream& operator<<(std::ostream& os, const L1GctHfEtSumsLut& lut);

  /// Set the function
  void setFunction(const L1CaloEtScale* const fn) {
    if (fn != nullptr) {
      m_lutFunction = fn;
      m_setupOk = true;
    }
  }

  /// Return the type of Lut
  L1GctHfEtSumsLut::hfLutType lutType() const { return m_lutType; }

  /// Return the Lut function
  const L1CaloEtScale* lutFunction() const { return m_lutFunction; }

  /// Get thresholds
  std::vector<double> getThresholdsGeV() const;
  std::vector<unsigned> getThresholdsGct() const;

protected:
  uint16_t value(const uint16_t lutAddress) const override;

private:
  const L1CaloEtScale* m_lutFunction;
  L1GctHfEtSumsLut::hfLutType m_lutType;
};

std::ostream& operator<<(std::ostream& os, const L1GctHfEtSumsLut& lut);

#endif /*L1GCTHFETSUMSLUT_H_*/
