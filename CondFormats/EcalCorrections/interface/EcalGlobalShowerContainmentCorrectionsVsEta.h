#ifndef CondFormats_EcalGlobalShowerContainmentCorrectionsVsEta_h
#define CondFormats_EcalGlobalShowerContainmentCorrectionsVsEta_h
// -*- C++ -*-
//
// Package:     CondFormats
// Class  :     EcalGlobalShowerContainmentCorrectionsVsEta
//
/**\class EcalGlobalShowerContainmentCorrectionsVsEta EcalGlobalShowerContainmentCorrectionsVsEta.h src/CondFormats/interface/EcalGlobalShowerContainmentCorrectionsVsEta.h

 * Description: Holds the coefficients of a polynomial that describes variation of the global containment effect as afunction of eta
 *
 * Usage example(for real code see 
 * CalibCalorimetry/EcalCorrectionModules/test) : 
 * \code

 ESHandle<EcalGlobalShowerContainmentCorrectionsVsEta> pGapCorr;
 iESetup.get<EcalGlobalShowerContainmentCorrectionsVsEtaRcd>().get(pGapCorr);
  
 double correction3x3 = pGapCorr->correction3x3(centerXtal,mathpoint);
 double correction5x5 = pGapCorr->correction5x5(centerXtal,mathpoint);


 * \endcode
 * \author       Paolo Meridiani
 * \id           $Id: EcalGlobalShowerContainmentCorrectionsVsEta.h,v 1.1 2007/07/13 17:37:06 meridian Exp $
*/

#include "CondFormats/Serialization/interface/Serializable.h"

#include <vector>
#include <algorithm>
#include <map>

class DetId;

class EcalGlobalShowerContainmentCorrectionsVsEta {
public:
  /// Structure defining the container for correction coefficients
  /**  data[0-2]    : 3x3
   *   data[3-5]    : 5x5
   */

  struct Coefficients {
    Coefficients() {
      for (unsigned int i = 0; i < Coefficients::kSize; ++i)
        data[i] = 0;
    }
    Coefficients(const Coefficients& coeff) { std::copy(coeff.data, coeff.data + Coefficients::kSize, data); }

    Coefficients& operator=(const Coefficients& coeff) {
      if (this == &coeff)
        return *this;
      std::copy(coeff.data, coeff.data + Coefficients::kSize, data);
      return *this;
    }

    ///The degree of the polynomial used as correction function plus one
    static const int kCoefficients = 3;

    ///Number of types of correction:  3x3, 5x5
    static const int kNTypes = 2;
    static const unsigned int kSize = kCoefficients * kNTypes;

    double data[kSize];

    COND_SERIALIZABLE;
  };

  /// The correction factor for 3x3 matrix
  /** @param pos is the distance in cm from the center of the xtal
   *  as calculated in RecoEcal/EgammaCoreTools/interface/PositionCalc.h 
   *  The valid return value is in the range (0,1] (divide by this
   *  value to apply the correction)
   *  Returns -1 if correction is not avaiable for that xtal*/
  const double correction3x3(const DetId& xtal) const;

  /// The correction factor for 5x5 matrix
  /** @param pos is the distance in cm from the center of the xtal
   *  as calculated in RecoEcal/EgammaCoreTools/interface/PositionCalc.h 
   *  The return value is in the range (0,1] (divide by this
   *  value to apply the correction)
   *  Returns -1 if correction is not avaiable for that xtal*/
  const double correction5x5(const DetId& xtal) const;

  /// Get the correction coefficients for the given xtal
  const Coefficients correctionCoefficients() const;

  /// Fill the correction coefficients
  void fillCorrectionCoefficients(const Coefficients& coefficients);

private:
  enum Type { e3x3, e5x5 };

  /** Calculate the correction for the given direction and type  */
  const double correction(const DetId& xtal, Type type) const;

  /// Holds the coeffiecients. The index corresponds to the group
  Coefficients coefficients_;

  COND_SERIALIZABLE;
};

#endif
