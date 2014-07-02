#ifndef RECOLOCALCALO_CALOTOWERSCREATOR_CALOTOWERSCREATIONALGO_H
#define RECOLOCALCALO_CALOTOWERSCREATOR_CALOTOWERSCREATIONALGO_H 1

#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

// channel status
#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"

#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"
#include "CondFormats/DataRecord/interface/HcalChannelQualityRcd.h"

// severity level assignment for HCAL
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSeverityLevelComputer.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSeverityLevelComputerRcd.h"

// severity level assignment for ECAL
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"

// need if we want to store the handles
#include "FWCore/Framework/interface/ESHandle.h"
#include <tuple>


#include <map>
class HcalTopology;
class CaloGeometry;
class CaloSubdetectorGeometry;
class CaloTowerConstituentsMap;
class CaloRecHit;
class DetId;

/** \class CaloTowersCreationAlgo
  *  
  * \author R. Wilkinson - Caltech
  */

//
// Modify MetaTower to save energy of rechits for use in tower 4-momentum assignment,
// added containers for timing assignment and for holding status information.
// Anton Anastassov (Northwestern)
//

class CaloTowersCreationAlgo {
public:

  int nalgo=-1;

  CaloTowersCreationAlgo();

  CaloTowersCreationAlgo(double EBthreshold, double EEthreshold, 

    bool useEtEBTreshold, bool useEtEETreshold,
    bool useSymEBTreshold, bool useSymEETreshold,				    

    double HcalThreshold,
    double HBthreshold, double HESthreshold, double HEDthreshold,
    double HOthreshold0, double HOthresholdPlus1, double HOthresholdMinus1,  
    double HOthresholdPlus2, double HOthresholdMinus2,
    double HF1threshold, double HF2threshold, 
    double EBweight, double EEweight,
    double HBweight, double HESweight, double HEDweight, 
    double HOweight, double HF1weight, double HF2weight,
    double EcutTower, double EBSumThreshold, double EESumThreshold, bool useHO,
    // (for momentum reconstruction algorithm)
    int momConstrMethod,
    double momHBDepth,
    double momHEDepth,
    double momEBDepth,
    double momEEDepth
    );
  
  CaloTowersCreationAlgo(double EBthreshold, double EEthreshold, 

    bool useEtEBTreshold, bool useEtEETreshold,
    bool useSymEBTreshold, bool useSymEETreshold,

    double HcalThreshold,
    double HBthreshold, double HESthreshold, double HEDthreshold,
    double HOthreshold0, double HOthresholdPlus1, double HOthresholdMinus1,  
    double HOthresholdPlus2, double HOthresholdMinus2, 
    double HF1threshold, double HF2threshold,
    const std::vector<double> & EBGrid, const std::vector<double> & EBWeights,
    const std::vector<double> & EEGrid, const std::vector<double> & EEWeights,
    const std::vector<double> & HBGrid, const std::vector<double> & HBWeights,
    const std::vector<double> & HESGrid, const std::vector<double> & HESWeights,
    const std::vector<double> & HEDGrid, const std::vector<double> & HEDWeights,
    const std::vector<double> & HOGrid, const std::vector<double> & HOWeights,
    const std::vector<double> & HF1Grid, const std::vector<double> & HF1Weights,
    const std::vector<double> & HF2Grid, const std::vector<double> & HF2Weights,
    double EBweight, double EEweight,
    double HBweight, double HESweight, double HEDweight, 
    double HOweight, double HF1weight, double HF2weight,
    double EcutTower, double EBSumThreshold, double EESumThreshold, bool useHO,
    // (for momentum reconstruction algorithm)
    int momConstrMethod,
    double momHBDepth,
    double momHEDepth,
    double momEBDepth,
    double momEEDepth
);
  
  void setGeometry(const CaloTowerConstituentsMap* cttopo, const HcalTopology* htopo, const CaloGeometry* geo);

  // pass the containers of channels status from the event record (stored in DB)
  // these are called in  CaloTowersCreator
  void setHcalChStatusFromDB(const HcalChannelQuality* s) { theHcalChStatus = s; }
  void setEcalChStatusFromDB(const EcalChannelStatus* s) { theEcalChStatus = s; }

  // Kake a map of number of channels not used in RecHit production.
  // The key is the calotower id.
  void makeHcalDropChMap();

  void makeEcalBadChs();

  void begin();
  void process(const HBHERecHitCollection& hbhe);
  void process(const HORecHitCollection& ho);
  void process(const HFRecHitCollection& hf); 
  void process(const EcalRecHitCollection& ecal); 
  
  
  void process(const CaloTowerCollection& ctc);

  void finish(CaloTowerCollection& destCollection);

  // modified rescale method
  void rescaleTowers(const CaloTowerCollection& ctInput, CaloTowerCollection& ctResult);

  void setEBEScale(double scale);
  void setEEEScale(double scale);
  void setHBEScale(double scale);
  void setHESEScale(double scale);
  void setHEDEScale(double scale);
  void setHOEScale(double scale);
  void setHF1EScale(double scale);
  void setHF2EScale(double scale);


  // Assign to categories based on info from DB and RecHit status
  // Called in assignHit to check if the energy should be added to
  // calotower, and how to flag the channel
  unsigned int hcalChanStatusForCaloTower(const CaloRecHit* hit);
  std::tuple<unsigned int,bool> ecalChanStatusForCaloTower(const EcalRecHit* hit);

  // Channel flagging is based on acceptable severity levels specified in the
  // configuration file. These methods are used to pass the values read in
  // CaloTowersCreator
  // 
  // from DB
  void setHcalAcceptSeverityLevel(unsigned int level) {theHcalAcceptSeverityLevel = level;} 
  void setEcalSeveritiesToBeExcluded(const std::vector<int>& ecalSev ) {theEcalSeveritiesToBeExcluded= ecalSev;} 

  // flag to use recovered hits
  void setRecoveredHcalHitsAreUsed(bool flag) {theRecoveredHcalHitsAreUsed = flag; };
  void setRecoveredEcalHitsAreUsed(bool flag) {theRecoveredEcalHitsAreUsed = flag; };

  //  severety level calculator for HCAL
  void setHcalSevLvlComputer(const HcalSeverityLevelComputer* c) {theHcalSevLvlComputer = c; };

  // severity level calculator for ECAL
  void setEcalSevLvlAlgo(const EcalSeverityLevelAlgo* a) { theEcalSevLvlAlgo =  a; }


  // The following are needed for creating towers from rechits excluded from the  ------------------------------------
  // default reconstructions

 // NB! Controls if rejected hits shold be used instead of the default!!!
  void setUseRejectedHitsOnly(bool flag) { useRejectedHitsOnly = flag; } 

  void setHcalAcceptSeverityLevelForRejectedHit(unsigned int level) {theHcalAcceptSeverityLevelForRejectedHit = level;} 
  //  void setEcalAcceptSeverityLevelForRejectedHit(unsigned int level) {theEcalAcceptSeverityLevelForRejectedHit = level;} 
  void SetEcalSeveritiesToBeUsedInBadTowers(const std::vector<int>& ecalSev ) {theEcalSeveritiesToBeUsedInBadTowers= ecalSev;} 


  void setUseRejectedRecoveredHcalHits(bool flag) {useRejectedRecoveredHcalHits = flag; };
  void setUseRejectedRecoveredEcalHits(bool flag) {useRejectedRecoveredEcalHits = flag; };

  //-------------------------------------------------------------------------------------------------------------------



  // set the EE EB handles
  
  void setEbHandle(const edm::Handle<EcalRecHitCollection> eb) { theEbHandle = eb; }
  void setEeHandle(const edm::Handle<EcalRecHitCollection> ee) { theEeHandle = ee; }




  // Add methods to get the seperate positions for ECAL/HCAL 
  // used in constructing the 4-vectors using new methods
  GlobalPoint emCrystalShwrPos (DetId detId, float fracDepth); 
  GlobalPoint hadSegmentShwrPos(DetId detId, float fracDepth);
  // "effective" point for the EM/HAD shower in CaloTower
  //  position based on non-zero energy cells
  GlobalPoint hadShwrPos(const std::vector<std::pair<DetId,float> >& metaContains,
    float fracDepth, double hadE);
  GlobalPoint emShwrPos(const std::vector<std::pair<DetId,float> >& metaContains, 
    float fracDepth, double totEmE);

  // overloaded function to get had position based on all had cells in the tower
  GlobalPoint hadShwrPos(CaloTowerDetId id, float fracDepth);
  GlobalPoint hadShwPosFromCells(DetId frontCell, DetId backCell, float fracDepth);

  // for Chris
  GlobalPoint emShwrLogWeightPos(const std::vector<std::pair<DetId,float> >& metaContains, 
    float fracDepth, double totEmE);


private:

  struct MetaTower {
    MetaTower(){}
    bool empty() const { return metaConstituents.empty();}
    // contains also energy of RecHit
    std::vector< std::pair<DetId, float> > metaConstituents;
    CaloTowerDetId id;
    float E=0, E_em=0, E_had=0, E_outer=0;
    float emSumTimeTimesE=0, hadSumTimeTimesE=0, emSumEForTime=0, hadSumEForTime=0; // Sum(Energy x Timing) : intermediate container

    // needed to set CaloTower status word
    int numBadEcalCells=0, numRecEcalCells=0, numProbEcalCells=0, numBadHcalCells=0, numRecHcalCells=0, numProbHcalCells=0; 

 };

  /// adds a single hit to the tower
  void assignHitEcal(const EcalRecHit* recHit);
  void assignHitHcal(const CaloRecHit* recHit);

  void rescale(const CaloTower * ct);

  /// looks for a given tower in the internal cache.  If it can't find it, it makes it.
  MetaTower & find(const CaloTowerDetId & id);
  
  /// helper method to look up the appropriate threshold & weight
  void getThresholdAndWeight(const DetId & detId, double & threshold, double & weight) const;

  double theEBthreshold, theEEthreshold;
  bool theUseEtEBTresholdFlag, theUseEtEETresholdFlag;
  bool theUseSymEBTresholdFlag,theUseSymEETresholdFlag;
  
  
  double  theHcalThreshold;

  double theHBthreshold, theHESthreshold,  theHEDthreshold; 
  double theHOthreshold0, theHOthresholdPlus1, theHOthresholdMinus1;
  double theHOthresholdPlus2, theHOthresholdMinus2, theHF1threshold, theHF2threshold;
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

  // for checking the status of ECAL and HCAL channels stored in the DB 
  const EcalChannelStatus* theEcalChStatus;
  const HcalChannelQuality* theHcalChStatus;

  // calculator of severety level for HCAL
  const HcalSeverityLevelComputer* theHcalSevLvlComputer;

  // calculator for severity level for ECAL
  const EcalSeverityLevelAlgo* theEcalSevLvlAlgo;

  
  // fields that hold the information passed from the CaloTowersCreator configuration file:
  // controll what is considered bad/recovered/problematic channel for CaloTower purposes 
  //
  unsigned int theHcalAcceptSeverityLevel;
  std::vector<int> theEcalSeveritiesToBeExcluded;
  // flag to use recovered hits
  bool theRecoveredHcalHitsAreUsed;
  bool theRecoveredEcalHitsAreUsed;

  // controls the tower reconstruction from rejected hits

  bool useRejectedHitsOnly;
  unsigned int theHcalAcceptSeverityLevelForRejectedHit;
  std::vector<int> theEcalSeveritiesToBeUsedInBadTowers;


  unsigned int useRejectedRecoveredHcalHits;
  unsigned int useRejectedRecoveredEcalHits;


  /// only affects energy and ET calculation.  HO is still recorded in the tower
  bool theHOIsUsed;

  // Switches and paramters for CaloTower 4-momentum assignment
  // "depth" variables do not affect all algorithms 
  int theMomConstrMethod;
  double theMomHBDepth;
  double theMomHEDepth;
  double theMomEBDepth;
  double theMomEEDepth;

  // compactify timing info
  int compactTime(float time);

  void convert(const CaloTowerDetId& id, const MetaTower& mt, CaloTowerCollection & collection);
  

  // internal map
  typedef std::vector<MetaTower> MetaTowerMap;
  MetaTowerMap theTowerMap;
  unsigned int theTowerMapSize=0;

  // Number of channels in the tower that were not used in RecHit production (dead/off,...).
  // These channels are added to the other "bad" channels found in the recHit collection. 
  typedef std::map<CaloTowerDetId, int> HcalDropChMap;
  HcalDropChMap hcalDropChMap;

  // Number of bad Ecal channel in each tower
  unsigned short ecalBadChs[CaloTowerDetId::kSizeForDenseIndexing];

  // clasification of channels in tower construction: the category definition is
  // affected by the setting in the configuration file
  // 
  enum ctHitCategory {GoodChan = 0, BadChan = 1, RecoveredChan = 2, ProblematicChan = 3, IgnoredChan = 99 };


  // the EE and EB collections for ecal anomalous cell info
   
  edm::Handle<EcalRecHitCollection> theEbHandle;
  edm::Handle<EcalRecHitCollection> theEeHandle;



};

#endif
