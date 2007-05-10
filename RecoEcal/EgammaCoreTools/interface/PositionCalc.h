#ifndef RecoEcal_EgammaCoreTools_PositionCalc_h
#define RecoEcal_EgammaCoreTools_PositionCalc_h

/** \class PositionClac
 *  
 * Finds the position and covariances for a cluster 
 * Formerly LogPositionCalc
 *
 * \author Ted Kolberg, ND
 * 
 * \version $Id: PositionCalc.h,v 1.8 2006/11/21 21:09:41 futyand Exp $
 *
 */

#include <vector>
#include <map>

#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

class PositionCalc
{
 public:
  // You must call Initialize before you can calculate positions or 
  // covariances.

  PositionCalc(std::map<std::string,double> providedParameters);
  PositionCalc() { };

  const PositionCalc& operator=(const PositionCalc& rhs);

  // Calculate_Location calculates an arithmetically or logarithmically
  // weighted average position of a vector of DetIds, which should be
  // a subset of the map used to Initialize.

  math::XYZPoint Calculate_Location(std::vector<DetId> passedDetIds,
                                    const EcalRecHitCollection *passedRecHitsMap,
                                    const CaloSubdetectorGeometry *passedGeometry,
                                    const CaloSubdetectorGeometry *passedGeometryES=0);

 private:
  bool        param_LogWeighted_;
  Double32_t  param_T0_barl_;
  Double32_t  param_T0_endc_;
  Double32_t  param_T0_endcPresh_;
  Double32_t  param_W0_;
  Double32_t  param_X0_;
  //const EcalRecHitCollection  *storedRecHitsMap_;
  //const CaloSubdetectorGeometry *storedSubdetectorGeometry_;

};

#endif
