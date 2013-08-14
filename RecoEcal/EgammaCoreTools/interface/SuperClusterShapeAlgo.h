#ifndef RecoEcal_EgammaCoreTools_SuperClusterShapeAlgo_h
#define RecoEcal_EgammaCoreTools_SuperClusterShapeAlgo_h

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

#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

class CaloSubdetectorTopology;

class SuperClusterShapeAlgo
{

 public:
  SuperClusterShapeAlgo(const EcalRecHitCollection* hits,
			const CaloSubdetectorGeometry* geometry);

  void Calculate_Covariances(const reco::SuperCluster &passedCluster);

  double etaWidth() { return etaWidth_; }
  double phiWidth() { return phiWidth_; }

 private:

  const EcalRecHitCollection* recHits_;
  const CaloSubdetectorGeometry* geometry_;

  double etaWidth_, phiWidth_;
  
};

#endif
