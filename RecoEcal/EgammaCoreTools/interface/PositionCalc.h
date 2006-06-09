#ifndef RecoEcal_EgammaCoreTools_PositionCalc_h
#define RecoEcal_EgammaCoreTools_PositionCalc_h

/** \class PositionClac
 *  
 * Finds the position and covariances for a cluster 
 * Formerly LogPositionCalc
 *
 * \author Ted Kolberg, ND
 * 
 * \version $Id: PositionCalc.h,v 1.3 2006/06/06 16:59:34 tkolberg Exp $
 *
 */

#include <vector>
#include <map>

#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

class PositionCalc
{
 public:
  // You must call Initialize before you can calculate positions or 
  // covariances.

  static void Initialize(std::map<std::string,double> providedParameters,
                         const std::map<DetId,EcalRecHit> *passedRecHitsMap,
                         const CaloSubdetectorGeometry *passedGeometry);  

  // Calculate_Location calculates an arithmetically or logarithmically
  // weighted average position of a vector of DetIds, which should be
  // a subset of the map used to Initialize.

  static math::XYZPoint Calculate_Location(std::vector<DetId> passedDetIds);

  // Calculate_Covariances calculates the variance in eta, variance in phi,
  // and covariance in eta and phi.  It must be given a vector of DetIds
  // which are a subset of the DetIds used to Initialize.

  static std::map<std::string,double> Calculate_Covariances(math::XYZPoint passedPoint,
                                                       std::vector<DetId> passedDetIds);

 private:
  PositionCalc();

  static bool        param_LogWeighted_;
  static Double32_t  param_X0_;
  static Double32_t  param_T0_;
  static Double32_t  param_W0_;
  static const std::map<DetId,EcalRecHit> *storedRecHitsMap_;
  static const CaloSubdetectorGeometry *storedSubdetectorGeometry_;

};

#endif
