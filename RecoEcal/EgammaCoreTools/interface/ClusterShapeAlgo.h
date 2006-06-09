#ifndef RecoEcal_EgammaCoreTools_ClusterShapeAlgo_h
#define RecoEcal_EgammaCoreTools_ClusterShapeAlgo_h

/** \class ClusterShapeAlgo
 *  
 * calculates and creates a ClusterShape object 
 *
 * \author Michael A. Balazs, UVa
 * 
 * \version $Id: ClusterShapeAlgo.h,v 1.5 2006/06/06 17:26:11 rahatlou Exp $
 *
 */


#include <map>

#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/EgammaReco/interface/ClusterShape.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

class ClusterShapeAlgo
{

 public:
  static void Initialize(const std::map<DetId,EcalRecHit> *passedRecHitsMap,
			 const edm::ESHandle<CaloGeometry> *geoHandle);
  static reco::ClusterShape Calculate(reco::BasicCluster passedCluster );
 
  private:
  ClusterShapeAlgo(){};

  void Calculate_TopEnergy(reco::BasicCluster passedCluster);
  void Calculate_2ndEnergy(reco::BasicCluster passedCluster);
  void Create_Map(); 
  void Calculate_e2x2();
  void Calculate_e3x2(); 
  void Calculate_e3x3();
  void Calculate_e5x5();
  void Calculate_Location(); 
  void Calculate_Covariances();
  
  static const edm::ESHandle<CaloGeometry> *storedGeoHandle_;
  static const std::map<DetId,EcalRecHit> *storedRecHitsMap_;

  std::pair<DetId, double> energyMap_[5][5];

  double covEtaEta_, covEtaPhi_, covPhiPhi_;
  double eMax_, e2nd_, e2x2_, e3x2_, e3x3_, e5x5_;
  double e3x2Ratio_;
  math::XYZPoint location_; 
  DetId eMaxId_, e2ndId_;

};

#endif
