#ifndef RecoEcal_EgammaCoreTools_ClusterShapeAlgo_h
#define RecoEcal_EgammaCoreTools_ClusterShapeAlgo_h

/** \class ClusterShapeAlgo
 *  
 * calculates and creates a ClusterShape object 
 *
 * \author Michael A. Balazs, UVa
 * 
 * \version $Id: ClusterShapeAlgo.h,v 1.11 2006/09/19 19:48:55 mabalazs Exp $
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
#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

class CaloSubdetectorTopology;

class ClusterShapeAlgo
{

 public:
  ClusterShapeAlgo(const PositionCalc& passedPositionCalc);
  ClusterShapeAlgo() { };
  reco::ClusterShape Calculate(const reco::BasicCluster &passedCluster,
                               const EcalRecHitCollection *hits,
                               const CaloSubdetectorGeometry * geometry,
                               const CaloSubdetectorTopology* topology);

  private:
  void Calculate_TopEnergy(const reco::BasicCluster &passedCluster,const EcalRecHitCollection *hits);
  void Calculate_2ndEnergy(const reco::BasicCluster &passedCluster,const EcalRecHitCollection *hits);
  void Create_Map(const EcalRecHitCollection *hits, const CaloSubdetectorTopology* topology);
  void Calculate_e2x2();
  void Calculate_e3x2();
  void Calculate_e3x3();
  void Calculate_e4x4();
  void Calculate_e5x5();
  void Calculate_Location(const EcalRecHitCollection *hits,const CaloSubdetectorGeometry * geometry);
  void Calculate_Covariances(const EcalRecHitCollection *hits,const CaloSubdetectorGeometry * geometry);
  void Calculate_BarrelBasketEnergyFraction(const reco::BasicCluster &passedCluster,const EcalRecHitCollection *hits,
                                            const int EtaPhi,const CaloSubdetectorGeometry * geometry);

  PositionCalc posCalculator_;

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
