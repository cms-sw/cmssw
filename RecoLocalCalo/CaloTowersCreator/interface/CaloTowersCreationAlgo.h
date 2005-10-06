#ifndef RECOLOCALCALO_CALOTOWERSCREATOR_CALOTOWERSCREATIONALGO_H
#define RECOLOCALCALO_CALOTOWERSCREATOR_CALOTOWERSCREATIONALGO_H 1

#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include <map>
class HcalTopology;
class CaloGeometry;
class CaloRecHit;
class DetId;

/** \class CaloTowersCreationAlgo
  *  
  * $Date: 2005/10/05 17:02:05 $
  * $Revision: 1.1 $
  * \author J. Mans - Minnesota
  */
class CaloTowersCreationAlgo {
public:
  CaloTowersCreationAlgo(const HcalTopology* topo, const CaloGeometry* geo);

  CaloTowersCreationAlgo(double EBthreshold, double EEthreshold, double HcalThreshold,
    double HBthreshold, double HESthreshold, double HEDthreshold,
    double HOthreshold, double HF1threshold, double HF2threshold,
    double EBweight, double EEweight, 
    double HBweight, double HESweight, double HEDweight, 
    double HOweight, double HF1weight, double HF2weight,
    double EcutTower, double EBSumThreshold, double EESumThreshold,
    const HcalTopology* topo, const CaloGeometry* geo, bool useHODefault);

  bool create(CaloTowerCollection& destCollection,
	      const HBHERecHitCollection& hbhe, 
	      const HORecHitCollection& ho, 
	      const HFRecHitCollection& hf); // eventually will need ECAL also.

private:
  /// adds a single hit to the tower
  void assignHit(const CaloRecHit * recHit);

  /// looks for a given tower in the internal cache.  If it can't find it, it makes it.
  CaloTower & find(CaloTowerDetId & id);

  /// helper method to look up the appropriate threshold & weight
  void getThresholdAndWeight(const DetId & detId, double & threshold, double & weight) const;

  double theEBthreshold, theEEthreshold, theHcalThreshold;
  double theHBthreshold, theHESthreshold,  theHEDthreshold; 
  double theHOthreshold, theHF1threshold, theHF2threshold;
  double theEBweight, theEEweight; 
  double theHBweight, theHESweight, theHEDweight, theHOweight, theHF1weight, theHF2weight;
  double theEcutTower, theEBSumThreshold, theEESumThreshold;

  const HcalTopology* theHcalTopology;
  const CaloGeometry* theGeometry;

  bool theHOIsUsedByDefault;

  // internal map
  typedef std::map<CaloTowerDetId, CaloTower *> CaloTowerMap;
  CaloTowerMap theCaloTowerMap;
};

#endif
