#ifndef RECOLOCALCALO_CALOTOWERSCREATOR_CALOTOWERSCREATIONALGO_H
#define RECOLOCALCALO_CALOTOWERSCREATOR_CALOTOWERSCREATIONALGO_H 1

#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include <map>
class HcalTopology;
class CaloGeometry;
class CaloSubdetectorGeometry;
class CaloTowerConstituentsMap;
class CaloRecHit;
class DetId;

/** \class CaloTowersCreationAlgo
  *  
  * $Date: 2007/03/31 18:38:51 $
  * $Revision: 1.9 $
  * \author R. Wilkinson - Caltech
  */

//
// Modify MetaTower to save energy of rechits for use in tower 4-momentum assignment
// Anton Anastassov (Northwestern)
//

class CaloTowersCreationAlgo {
public:
  CaloTowersCreationAlgo();

  CaloTowersCreationAlgo(double EBthreshold, double EEthreshold, double HcalThreshold,
    double HBthreshold, double HESthreshold, double HEDthreshold,
    double HOthreshold, double HF1threshold, double HF2threshold,
    double EBweight, double EEweight,
    double HBweight, double HESweight, double HEDweight, 
    double HOweight, double HF1weight, double HF2weight,
    double EcutTower, double EBSumThreshold, double EESumThreshold, bool useHO,
    // (for momentum reconstruction algorithm)
    int momConstrMethod,
    double momEmDepth,
    double momHadDepth,
    double momTotDepth
    );
  
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
    double EcutTower, double EBSumThreshold, double EESumThreshold, bool useHO,
    // (for momentum reconstruction algorithm)
    int momConstrMethod,
    double momEmDepth,
    double momHadDepth,
    double momTotDepth
    );
  
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

   // set momentum construction method and parameters
  void setMomConstrMethod(int methodId);
  void setMomEmDepth(double momEmDepth);
  void setMomHadDepth(double momHadDepth);
  void setMomTotDepth(double momTotDepth);

  // Add methods to get the seperate positions for ECAL/HCAL 
  // used in constructing the 4-vectors using new methods
  GlobalPoint emCrystalShwrPos (DetId detId, float fracDepth); 
  GlobalPoint hadSegmentShwrPos(DetId detId, float fracDepth);
  // "effective" point for the EM/HAD shower in CaloTower
  GlobalPoint hadShwrPos(std::vector<std::pair<DetId,double> >& metaContains,
    float fracDepth, double hadE);
  GlobalPoint emShwrPos(std::vector<std::pair<DetId,double> >& metaContains, 
    float fracDepth, double totEmE);

private:

  struct MetaTower {
    MetaTower();
    double E, E_em, E_had, E_outer;
    // contains also energy of RecHit
    std::vector< std::pair<DetId, double> > metaConstituents;
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

  // Switches and paramters for CaloTower 4-momentum assignment
  // "depth" variables do not affect all algorithms 
  int theMomConstrMethod;
  double theMomEmDepth;
  double theMomHadDepth;
  double theMomTotDepth;

  CaloTower convert(const CaloTowerDetId& id, const MetaTower& mt);

  // internal map
  typedef std::map<CaloTowerDetId, MetaTower> MetaTowerMap;
  MetaTowerMap theTowerMap;
};

#endif
