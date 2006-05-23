#ifndef RecoEcal_EgammaCoreTools_ClusterShapeAlgo_h
#define RecoEcal_EgammaCoreTools_ClusterShapeAlgo_h

/** \class ClusterShapeAlgo
 *  
 * calculates and creates a ClusterShape object 
 *
 * \author Michael A. Balazs, UVa
 * 
 * \version $Id: ClusterShape.h,v 1.3  2006/05/17 12:00:00 mbalazs Exp $
 * \version $Id: ClusterShape.h,v 1.2  2006/05/15 12:00:00 mbalazs Exp $
 * \version $Id: ClusterShape.h,v 1.1  2006/05/11 12:00:00 mbalazs Exp $
 * \version $Id: ClusterShape.h,v 1.0  2006/05/05 12:00:00 mbalazs Exp $
 *
 */


#include <map>

#include "DataFormats/EgammaReco/interface/ClusterShape.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/Math/interface/Point3D.h"

class ClusterShapeAlgo
{

 public:
  static void Initialize(const std::map<EBDetId,EcalRecHit> *passedRecHitsMap,
			 std::string passedCollectionType);
  static reco::ClusterShape Calculate(reco::BasicCluster passedCluster );
 
  // private:
  ClusterShapeAlgo(){};

  void Calculate_TopEnergy(reco::BasicCluster passedCluster);
  void Calculate_2ndEnergy(reco::BasicCluster passedCluster);
  void Create_Map(); // Will need to add endcap and preshower code in here
  void Calculate_e2x2();
  void Calculate_e3x2(); 
  void Calculate_e3x3();
  void Calculate_e5x5();
  void Calculate_Location(); //To Be Completed Pending Position Calc
  void Calculate_Covariances(); //To Be Completed Pending Position Calc
  
  static std::string param_CollectionType_;
  static const std::map<EBDetId,EcalRecHit> *storedRecHitsMap_;

  std::pair<EBDetId, Double32_t> energyMap_[5][5];

  Double32_t covEtaEta_, covEtaPhi_, covPhiPhi_;
  Double32_t eMax_, e2nd_, e2x2_, e3x2_, e3x3_, e5x5_;
  Double32_t e3x2Ratio_;
  math::XYZPoint location_; 
  DetId eMaxId_, e2ndId_;

};

#endif
