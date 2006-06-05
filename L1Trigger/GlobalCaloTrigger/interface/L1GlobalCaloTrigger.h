#ifndef L1GLOBALCALOTRIGGER_H_
#define L1GLOBALCALOTRIGGER_H_

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctDigis.h"

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetCand.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctEtTypes.h"

#include <vector>



/**
  * Represents the GCT system
  * This is the main point of access for the user
  * 
  * author: Jim Brooke
  * date: 20/2/2006
  * 
  **/ 

class L1GctSourceCard;
class L1GctJetLeafCard;
class L1GctEmLeafCard;

class L1GctWheelJetFpga;
class L1GctWheelEnergyFpga;
class L1GctJetFinalStage;
class L1GctGlobalEnergyAlgos;
class L1GctElectronFinalSort;
class L1GctJetEtCalibrationLut;


class L1GlobalCaloTrigger {
public:
  /// Declare numbers of each card type
  static const int N_SOURCE_CARDS;
  static const int N_JET_LEAF_CARDS;
  static const int N_EM_LEAF_CARDS;
  static const int N_WHEEL_CARDS;
  
  /// construct the GCT
  L1GlobalCaloTrigger(bool useFile=false);
  ///
  /// dismantle the GCT
  ~L1GlobalCaloTrigger();
  ///
  /// load files into Source Cards
  void openSourceCardFiles(std::string fileBase);
  
  /// Reset internal buffers
  void reset();
  
  /// process an event
  void process();
  
  /// iso electron outputs to GT
  std::vector<L1GctEmCand> getIsoElectrons();
  
  /// non-iso electron outputs to GT
  std::vector<L1GctEmCand> getNonIsoElectrons();
  
  /// central jet outputs to GT
  std::vector<L1GctJetCand> getCentralJets();
  
  /// forward jet outputs to GT
  std::vector<L1GctJetCand> getForwardJets();
  
  /// tau jet outputs to GT
  std::vector<L1GctJetCand> getTauJets();
  
  /// Total Et output to GT
  L1GctScalarEtVal getEtSum();
  
  /// Total hadronic Et output to GT
  L1GctScalarEtVal getEtHad();

  /// Etmiss output to GT
  L1GctScalarEtVal getEtMiss();
  
  /// Etmiss phi output to GT
  L1GctEtAngleBin getEtMissPhi();

  // Jet Count output to GT
  L1GctJcFinalType getJetCount(unsigned jcnum);

  /// get the Source cards
  std::vector<L1GctSourceCard*> getSourceCards() { return theSourceCards; }
  
  /// get the Jet Leaf cards
  std::vector<L1GctJetLeafCard*> getJetLeafCards() { return theJetLeafCards; }
  
  /// get the Jet Leaf cards
  std::vector<L1GctEmLeafCard*> getEmLeafCards() { return theEmLeafCards; }
  
  /// get the Wheel Jet FPGAs
  std::vector<L1GctWheelJetFpga*> getWheelJetFpgas() { return theWheelJetFpgas; }
  
  /// get the Wheel Energy Fpgas
  std::vector<L1GctWheelEnergyFpga*> getWheelEnergyFpgas() { return theWheelEnergyFpgas; }
  
  /// get the jet final stage
  L1GctJetFinalStage* getJetFinalStage() { return theJetFinalStage; }
  
  /// get the energy final stage
  L1GctGlobalEnergyAlgos* getEnergyFinalStage() { return theEnergyFinalStage; }
  
  /// get the electron final stage sorters
  L1GctElectronFinalSort* getIsoEmFinalStage() { return theIsoEmFinalStage; }
  L1GctElectronFinalSort* getNonIsoEmFinalStage() { return theNonIsoEmFinalStage; }

  /// get the Jet Et calibration LUT
  L1GctJetEtCalibrationLut* getJetEtCalibLut() { return m_jetEtCalLut; }

  /// print setup info
  void print();
  
 private:
  
  /// instantiate the hardware & algo objects and wire up the system
  void build();
  
 private:
  
  /// where are we getting data from?
  bool readFromFile;

  /// pointers to the Source Cards
  std::vector<L1GctSourceCard*> theSourceCards;
  
  /// pointers to the Jet Leaf cards
  std::vector<L1GctJetLeafCard*> theJetLeafCards;
  
  /// pointers to the EM Leaf cards
  std::vector<L1GctEmLeafCard*> theEmLeafCards;
  
  /// Wheel Card Jet Fpgas	
  std::vector<L1GctWheelJetFpga*> theWheelJetFpgas;		
  
  /// Wheel Card Energy Fpgas
  std::vector<L1GctWheelEnergyFpga*> theWheelEnergyFpgas;
  
  /// jet final stage algo
  L1GctJetFinalStage* theJetFinalStage;			
  
  /// energy final stage algos
  L1GctGlobalEnergyAlgos* theEnergyFinalStage;	
  
  /// electron final stage sorters
  L1GctElectronFinalSort* theIsoEmFinalStage;
  L1GctElectronFinalSort* theNonIsoEmFinalStage;

  /// Jet Et calibraion LUT
  L1GctJetEtCalibrationLut* m_jetEtCalLut;
  
};

#endif /*L1GLOBALCALOTRIGGER_H_*/
