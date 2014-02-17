#ifndef RecoEcal_EgammaCoreTools_PositionCalc_h
#define RecoEcal_EgammaCoreTools_PositionCalc_h

/** \class PositionClac
 *  
 * Finds the position and covariances for a cluster 
 * Formerly LogPositionCalc
 *
 * \author Ted Kolberg, ND
 * 
 * \version $Id: PositionCalc.h,v 1.14 2010/11/16 15:09:27 argiro Exp $
 *
 */

#include <vector>
#include <map>

#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

class PositionCalc
{
 public:
  // You must call Initialize before you can calculate positions or 
  // covariances.

  PositionCalc(const edm::ParameterSet& par);
  PositionCalc() { };

  const PositionCalc& operator=(const PositionCalc& rhs);

  // Calculate_Location calculates an arithmetically or logarithmically
  // weighted average position of a vector of DetIds, which should be
  // a subset of the map used to Initialize.

  math::XYZPoint Calculate_Location( const std::vector< std::pair< DetId, float > >&      iDetIds  ,
				     const EcalRecHitCollection*    iRecHits ,
				     const CaloSubdetectorGeometry* iSubGeom ,
				     const CaloSubdetectorGeometry* iESGeom = 0 ) ;

 private:
  bool    param_LogWeighted_;
  double  param_T0_barl_;
  double  param_T0_endc_;
  double  param_T0_endcPresh_;
  double  param_W0_;
  double  param_X0_;

  const CaloSubdetectorGeometry* m_esGeom ;
  bool m_esPlus ;
  bool m_esMinus ;

};

#endif
