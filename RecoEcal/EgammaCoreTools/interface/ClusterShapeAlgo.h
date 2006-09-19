#ifndef RecoEcal_EgammaCoreTools_ClusterShapeAlgo_h
#define RecoEcal_EgammaCoreTools_ClusterShapeAlgo_h

/** \class ClusterShapeAlgo
 *  
 * calculates and creates a ClusterShape object 
 *
 * \author Michael A. Balazs, UVa
 * 
 * \version $Id: ClusterShapeAlgo.h,v 1.9 2006/09/16 01:44:31 mabalazs Exp $
 *
 */

#include <map>

#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/EgammaReco/interface/ClusterShape.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

class ClusterShapeAlgo
{

 public:
  static void Initialize(const EcalRecHitCollection *passedRecHitsMap,
			 const edm::ESHandle<CaloGeometry> *geoHandle);
  static reco::ClusterShape Calculate(const reco::BasicCluster &passedCluster );
 
  private:
  ClusterShapeAlgo(){};

  void Calculate_TopEnergy(const reco::BasicCluster &passedCluster);
  void Calculate_2ndEnergy(const reco::BasicCluster &passedCluster);
  void Create_Map(); 
  void Calculate_e2x2();
  void Calculate_e3x2(); 
  void Calculate_e3x3();
  void Calculate_e4x4();
  void Calculate_e5x5();
  void Calculate_Location(); 
  void Calculate_Covariances();
  void Calculate_BarrelBasketEnergyFraction(const reco::BasicCluster &passedCluster, const int EtaPhi);
  
  static const edm::ESHandle<CaloGeometry> *storedGeoHandle_;
  static const EcalRecHitCollection *storedRecHitsMap_;

  std::pair<DetId, double> energyMap_[5][5];
  int e2x2_Diagonal_X_, e2x2_Diagonal_Y_; 

  double covEtaEta_, covEtaPhi_, covPhiPhi_;
  double eMax_, e2nd_, e2x2_, e3x2_, e3x3_, e4x4_, e5x5_;
  double e3x2Ratio_;
  std::vector<double> energyBasketFractionEta_;
  std::vector<double> energyBasketFractionPhi_;
  math::XYZPoint location_; 
  DetId eMaxId_, e2ndId_;
  
  enum { Eta, Phi };

};

#endif
