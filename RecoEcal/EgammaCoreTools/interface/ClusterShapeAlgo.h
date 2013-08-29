#ifndef RecoEcal_EgammaCoreTools_ClusterShapeAlgo_h
#define RecoEcal_EgammaCoreTools_ClusterShapeAlgo_h

/** \class ClusterShapeAlgo
 *  
 * calculates and creates a ClusterShape object 
 *
 * \author Michael A. Balazs, UVa
 * 
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
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class CaloSubdetectorTopology;

struct EcalClusterEnergyDeposition
{ 
  double deposited_energy;
  double r;
  double phi;
};

class ClusterShapeAlgo
{

 public:
  ClusterShapeAlgo(const edm::ParameterSet& par);
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
  void Calculate_e2x5Right();
  void Calculate_e2x5Left();
  void Calculate_e2x5Top();
  void Calculate_e2x5Bottom();
  void Calculate_Covariances(const reco::BasicCluster &passedCluster,
			     const EcalRecHitCollection* hits,
			     const CaloSubdetectorGeometry* geometry);
  void Calculate_BarrelBasketEnergyFraction(const reco::BasicCluster &passedCluster,const EcalRecHitCollection *hits,
                                            const int EtaPhi,const CaloSubdetectorGeometry * geometry);
  // defines a energy deposition topology in a reference system centered on the cluster
  void Calculate_EnergyDepTopology(const reco::BasicCluster &passedCluster,const EcalRecHitCollection *hits, const CaloSubdetectorGeometry * geometry, bool logW=true);
  void Calculate_Polynomials(double rho);
  double factorial(int n) const;
  void Calculate_lat(const reco::BasicCluster &passedCluster);
  void Calculate_ComplexZernikeMoments(const reco::BasicCluster &passedCluster);
  // explicit implementation of polynomial part of
  // Zernike-Functions for n<=5;
  double f00(double r);
  double f11(double r);
  double f20(double r);
  double f22(double r);
  double f31(double r);
  double f33(double r);
  double f40(double r);
  double f42(double r);
  double f44(double r);
  double f51(double r);
  double f53(double r);
  double f55(double r);
  double absZernikeMoment(const reco::BasicCluster &passedCluster, int n, int m, double R0=6.6);
  double fast_AbsZernikeMoment(const reco::BasicCluster &passedCluster, int n, int m, double R0);
  // Calculation of Zernike-Moments for general values of (n,m)
  double calc_AbsZernikeMoment(const reco::BasicCluster &passedCluster, int n, int m, double R0);

  edm::ParameterSet parameterSet_;

  std::pair<DetId, double> energyMap_[5][5];
  int e2x2_Diagonal_X_, e2x2_Diagonal_Y_;

  double covEtaEta_, covEtaPhi_, covPhiPhi_;
  double eMax_, e2nd_, e2x2_, e3x2_, e3x3_, e4x4_, e5x5_;
  double e2x5Right_, e2x5Left_, e2x5Top_, e2x5Bottom_;
  double e3x2Ratio_;
  double lat_;
  double etaLat_ ;
  double phiLat_ ;
  double A20_, A42_;
  std::vector<double> energyBasketFractionEta_;
  std::vector<double> energyBasketFractionPhi_;
  DetId eMaxId_, e2ndId_;
  std::vector<EcalClusterEnergyDeposition> energyDistribution_;
  std::vector<double> fcn_;

  enum { Eta, Phi };

};

#endif
