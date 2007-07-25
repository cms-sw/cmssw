#ifndef CondFormats_EcalShowerContainmentCorrectionsLogE2E1_h
#define CondFormats_EcalShowerContainmentCorrectionsLogE2E1_h
// -*- C++ -*-
//
// Package:     CondFormats
// Class  :     EcalShowerContainmentCorrectionsLogE2E1
// 
/**\class EcalShowerContainmentCorrectionsLogE2E1 EcalShowerContainmentCorrectionsLogE2E1.h src/CondFormats/interface/EcalShowerContainmentCorrectionsLogE2E1.h

 * Description: Holds the coefficients of a polynomial that describes
 *              the shower containment as a function of LogE2/E1.
 *              See CMS Internal Note 2006/016
 * 
 * Usage example(for real code see 
 * CalibCalorimetry/EcalCorrectionModules/test) : 
 * \code

 ESHandle<EcalShowerContainmentCorrectionsLogE2E1> pGapCorr;
 iESetup.get<EcalShowerContainmentCorrectionsLogE2E1Rcd>().get(pGapCorr);
  
 PositionCalc pos(...);
 
 Hep3Vector clusterPos= pos.CalculatePos(...);	      
 math::XYZPoint mathpoint(clusterPos.x(),clusterPos.y(),clusterPos.z());
 
 double correction3x3 = pGapCorr->correction3x3(centerXtal,mathpoint);
 double correction5x5 = pGapCorr->correction5x5(centerXtal,mathpoint);


 * \endcode

 * \author       Stefano Argiro'
 * \date         Fri Mar  2 16:50:49 CET 2007
 * \id           $Id: EcalShowerContainmentCorrectionsLogE2E1.h,v 1.1 2007/05/15 20:37:22 argiro Exp $
*/

#include <vector>
#include <algorithm>
#include <map>
#include <boost/cstdint.hpp>

#include <DataFormats/Math/interface/Point3D.h>

class EBDetId;

class EcalShowerContainmentCorrectionsLogE2E1 {

 public:

  /// Structure defining the container for correction coefficients
  /**  data[0-3]    : 3x3, x, right
   *   data[4-7]    : 3x3, x, left
   *   data[8-11]   : 3x3, y, right
   *   data[12-15]  : 3x3, y, left
   *   data[16-19]  : 5x5, x, right
   *   data[20-23]  : 5x5, x, left
   *   data[24-27]  : 5x5, y, right
   *   data[28-31]  : 5x5, y, left
   */

  struct Coefficients{

    Coefficients(){for(unsigned int i=0; i<Coefficients::kSize; ++i) data[i]=0;}
    Coefficients(const Coefficients& coeff){
      std::copy(coeff.data,
		coeff.data+Coefficients::kSize,
		data);
    }
 
    Coefficients& operator=(const Coefficients& coeff){
      if (this == &coeff) return *this; 
      std::copy(coeff.data,coeff.data+Coefficients::kSize,data);
      return *this;
    }

    ///The degree of the polynomial used as correction function plus one
    static const int kPolynomialDegree = 4;

    ///Number of types of correction:  Left, right, 3x3, 5x5, x, y
    static const int kNTypes           = 8; 
    static const unsigned int kSize    = kPolynomialDegree* kNTypes  ;

    double data[kSize];

  };


  /// Get the correction coefficients for the given xtal
  /** Return zero coefficients in case the correction is not available
      for that xtal */
  const Coefficients correctionCoefficients(const EBDetId& centerxtal) const;

  /// Fill the correction coefficients for a given xtal, part of group @group
  /** Do not replace if xtal is already there*/
  void fillCorrectionCoefficients(const EBDetId& xtal, 
                                  int   group, 
				  const Coefficients&  coefficients);

  /// Fill the correction coefficients for a given Ecal module
  /** Assume that corresponding  modules in different supermodules use
      the same coefficients*/
  void fillCorrectionCoefficients(const int supermodule, const int module, 
				  const Coefficients&  coefficients);


  /// The correction factor for 3x3 matrix
  /** @param pos is the distance in cm from the center of the xtal
   *  as calculated in RecoEcal/EgammaCoreTools/interface/PositionCalc.h 
   *  The valid return value is in the range (0,1] (divide by this
   *  value to apply the correction)
   *  Returns -1 if correction is not avaiable for that xtal*/
  const double correction3x3(const EBDetId& xtal, 
			     const double & loge2e1_eta,
			     const double & loge2e1_phi) const;


 /// The correction factor for 5x5 matrix
  /** @param pos is the distance in cm from the center of the xtal
   *  as calculated in RecoEcal/EgammaCoreTools/interface/PositionCalc.h 
   *  The return value is in the range (0,1] (divide by this
   *  value to apply the correction)
   *  Returns -1 if correction is not avaiable for that xtal*/
  const double correction5x5(const EBDetId& xtal, 
			     const double & loge2e1_eta,
			     const double & loge2e1_phi) const;



 private:
  enum  Direction{eEta,ePhi};
  enum  Type{e3x3,e5x5};

  /** Calculate the correction for the given direction and type  */
  const double correctionEtaPhi(const EBDetId& xtal, 
			    double position, 
			    Direction dir,
			    Type type) const ;


  typedef std::map<EBDetId,int> GroupMap;

  /// Maps in which group a particular xtal has been placed
  GroupMap     groupmap_;


  /// Holds the coeffiecients. The index corresponds to the group
  std::vector<Coefficients> coefficients_;  

};

#endif
