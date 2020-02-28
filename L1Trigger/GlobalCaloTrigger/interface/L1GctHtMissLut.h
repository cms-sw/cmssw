#ifndef L1GCTHTMISSLUT_H_
#define L1GCTHTMISSLUT_H_

#include "L1Trigger/GlobalCaloTrigger/src/L1GctLut.h"

#include <vector>

/*!
 * \author Greg Heath
 * \date September 2008
 */

/*! \class L1GctHtMissLut
 * \brief LUT for conversion of Ht components x and y to magnitude and angle
 *
 */

class L1CaloEtScale;

class L1GctHtMissLut : public L1GctLut<16, 12>

{
public:
  enum numberOfBits { kHxOrHyMissComponentNBits = 8, kHtMissMagnitudeNBits = 7, kHtMissAngleNBits = 5 };

  // Definitions.
  static const int NAddress, NData;

  /// Constructor for use with emulator
  L1GctHtMissLut(const L1CaloEtScale* const scale, const double lsb);
  /// Default constructor
  L1GctHtMissLut();
  /// Copy constructor
  L1GctHtMissLut(const L1GctHtMissLut& lut);
  /// Destructor
  ~L1GctHtMissLut() override;

  /// Overload = operator
  L1GctHtMissLut operator=(const L1GctHtMissLut& lut);

  /// Overload << operator
  friend std::ostream& operator<<(std::ostream& os, const L1GctHtMissLut& lut);

  /// Set the functions
  void setEtScale(const L1CaloEtScale* const fn) {
    m_etScale = fn;
    if (fn != nullptr) {
      m_setupOk = true;
    }
  }
  void setExEyLsb(const double lsb) { m_componentLsb = lsb; }

  /// Return the Lut functions and parameters
  const L1CaloEtScale* etScale() const { return m_etScale; }
  const double componentLsb() const { return m_componentLsb; }

  /// Get thresholds
  std::vector<double> getThresholdsGeV() const;
  std::vector<unsigned> getThresholdsGct() const;

protected:
  uint16_t value(const uint16_t lutAddress) const override;

private:
  const L1CaloEtScale* m_etScale;

  double m_componentLsb;
};

std::ostream& operator<<(std::ostream& os, const L1GctHtMissLut& lut);

#endif /*L1GCTHTMISSLUT_H_*/
