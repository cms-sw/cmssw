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
  * $Date: 2006/05/11 20:57:20 $
  * $Revision: 1.8 $
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
  
  CaloTowersCreationAlgo(double EBthreshold, double EEthreshold, double HcalThreshold,
    double HBthreshold, double HESthreshold, double HEDthreshold,
    double HOthreshold, double HF1threshold, double HF2threshold,
    std::vector<double> EBGrid, std::vector<double> EBWeights,
    std::vector<double> EEGrid, std::vector<double> EEWeights,
    std::vector<double> HBGrid, std::vector<double> HBWeights,
    std::vector<double> HESGrid, std::vector<double> HESWeights,
    std::vector<double> HEDGrid, std::vector<double> HEDWeights,
    std::vector<double> HOGrid, std::vector<double> HOWeights,
    std::vector<double> HF1Grid, std::vector<double> HF1Weights,
    std::vector<double> HF2Grid, std::vector<double> HF2Weights,
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
  void process(const CaloTowerCollection& ctc);

  void finish(CaloTowerCollection& destCollection);
  void setEBEScale(double scale);
  void setEEEScale(double scale);
  void setHBEScale(double scale);
  void setHESEScale(double scale);
  void setHEDEScale(double scale);
  void setHOEScale(double scale);
  void setHF1EScale(double scale);
  void setHF2EScale(double scale);

private:
  struct MetaTower {
    MetaTower();
    double E, E_em, E_had, E_outer;
    std::vector<DetId> constituents;
  };

  /// adds a single hit to the tower
  void assignHit(const CaloRecHit * recHit);
 
  void rescale(const CaloTower * ct);
  /// looks for a given tower in the internal cache.  If it can't find it, it makes it.
  MetaTower & find(const CaloTowerDetId & id);
  
  /// helper method to look up the appropriate threshold & weight
  void getThresholdAndWeight(const DetId & detId, double & threshold, double & weight) const;
  
  double theEBthreshold, theEEthreshold, theHcalThreshold;
  double theHBthreshold, theHESthreshold,  theHEDthreshold; 
  double theHOthreshold, theHF1threshold, theHF2threshold;
  std::vector<double> theEBGrid, theEBWeights;
  std::vector<double> theEEGrid, theEEWeights;
  std::vector<double> theHBGrid, theHBWeights;
  std::vector<double> theHESGrid, theHESWeights;
  std::vector<double> theHEDGrid, theHEDWeights;
  std::vector<double> theHOGrid, theHOWeights;
  std::vector<double> theHF1Grid, theHF1Weights;
  std::vector<double> theHF2Grid, theHF2Weights;
  double theEBweight, theEEweight;
  double theHBweight, theHESweight, theHEDweight, theHOweight, theHF1weight, theHF2weight;
  double theEcutTower, theEBSumThreshold, theEESumThreshold;

  double theEBEScale;
  double theEEEScale;
  double theHBEScale;
  double theHESEScale;
  double theHEDEScale;
  double theHOEScale;
  double theHF1EScale;
  double theHF2EScale;
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
