#ifndef RecoEcal_EgammaCoreTools_ClusterShapeAlgo_h
#define RecoEcal_EgammaCoreTools_ClusterShapeAlgo_h

/** \class ClusterShapeAlgo
 *  
 * calculates and creates a ClusterShape object 
 *
 * \author Michael A. Balazs, UVa
 * 
 * \version $Id: ClusterShape.h,v 1.2 2006/05/15 12:00:00 mbalazs Exp $
 * \version $Id: ClusterShape.h,v 1.1 2006/05/11 12:00:00 mbalazs Exp $
 * \version $Id: ClusterShape.h,v 1.0 2006/05/05 12:00:00 mbalazs Exp $
 *
 */


#include <map>

#include "DataFormats/EgammaReco/interface/ClusterShape.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"

class ClusterShapeAlgo
{

 public:
  static void Initialize(std::map<std::string,double> providedParameters,
			 std::map<EBDetId,EcalRecHit> *passedRecHitsMap);
  static reco::ClusterShape Calculate(reco::BasicCluster passedCluster );
 
  // private:
  ClusterShapeAlgo(){};

  void Calculate_TopEnergy(reco::BasicCluster passedCluster);
  void Calculate_2ndEnergy(reco::BasicCluster passedCluster);
  void Create_Map(); // do i need to worry about barrel vs endcap here?
  void Calculate_e2x2();
  void Calculate_e3x2(); // does this do the hadoverecal (=ratio);
  void Calculate_e3x3();
  void Calculate_e5x5();
  void Calculate_Weights();
  void Calculate_eta25phi25();
  void Calculate_Covariances();
  void Calculate_hadOverEcal(); // is this needed? see above

  static bool       param_LogWeighted_;
  static Double32_t param_X0_;
  static Double32_t param_T0_;
  static Double32_t param_W0_;

  static std::map<EBDetId,EcalRecHit> *storedRecHitsMap_;

  std::pair<DetId, Double32_t> energyMap_[5][5]; // maybe only energy needed... see if realy needed for the eta25/phi25
  Double32_t weightsMap_[5][5];
  Double32_t weightsTotal_;

  Double32_t covEtaEta_, covEtaPhi_, covPhiPhi_;
  Double32_t eMax_, e2nd_, e2x2_, e3x2_, e3x3_, e5x5_;
  Double32_t eta25_, phi25_; // this may be done in LogPositionCalc and can i get that data out?
  Double32_t hadOverEcal_;
  DetId eMaxId_, e2ndId_;

};

#endif
