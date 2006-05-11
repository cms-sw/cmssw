#ifndef RECOLOCALCALO_CALOTOWERSCREATOR_CALOTOWERSCREATIONALGO_H
#define RECOLOCALCALO_CALOTOWERSCREATOR_CALOTOWERSCREATIONALGO_H 1

#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include <map>
class HcalTopology;
class CaloGeometry;
class CaloSubdetectorGeometry;
class CaloTowerConstituentsMap;
class CaloRecHit;
class DetId;

/** \class CaloTowersCreationAlgo
  *  
  * $Date: 2006/03/29 21:25:47 $
  * $Revision: 1.7 $
  * \author R. Wilkinson - Caltech
  */
class CaloTowersCreationAlgo {
public:
  CaloTowersCreationAlgo();

  CaloTowersCreationAlgo(double EBthreshold, double EEthreshold, double HcalThreshold,
    double HBthreshold, double HESthreshold, double HEDthreshold,
    double HOthreshold, double HF1threshold, double HF2threshold,
    double EBweight, double EEweight, 
    double HBweight, double HESweight, double HEDweight, 
    double HOweight, double HF1weight, double HF2weight,
    double EcutTower, double EBSumThreshold, double EESumThreshold, bool useHO);
  
  void setGeometry(const CaloTowerConstituentsMap* cttopo, const HcalTopology* htopo, const CaloGeometry* geo);

  void begin();
  void process(const HBHERecHitCollection& hbhe);
  void process(const HORecHitCollection& ho);
  void process(const HFRecHitCollection& hf); 
  void process(const EcalRecHitCollection& ecal); 

  void finish(CaloTowerCollection& destCollection);

private:
  struct MetaTower {
    MetaTower();
    double E, E_em, E_had, E_outer;
    std::vector<DetId> constituents;
  };

  /// adds a single hit to the tower
  void assignHit(const CaloRecHit * recHit);
  
  /// looks for a given tower in the internal cache.  If it can't find it, it makes it.
  MetaTower & find(const CaloTowerDetId & id);
  
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
  const CaloTowerConstituentsMap* theTowerConstituentsMap;
  const CaloSubdetectorGeometry* theTowerGeometry;

  /// only affects energy and ET calculation.  HO is still recorded in the tower
  bool theHOIsUsed;


  CaloTower convert(const CaloTowerDetId& id, const MetaTower& mt);

  // internal map
  typedef std::map<CaloTowerDetId, MetaTower> MetaTowerMap;
  MetaTowerMap theTowerMap;
};

#endif
