#ifndef L1GLOBALCALOTRIGGER_H_
#define L1GLOBALCALOTRIGGER_H_

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctDigis.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctRegion.h"

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetCand.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctEtTypes.h"

#include <vector>

/*!
 * \author Jim Brooke
 * \date Feb 2006
 */

/*!
 * \class L1GlobalCaloTrigger
 * \brief Bit-level emulation of the Global Calorimeter Trigger
 * 
 * 
 */


class L1GctSourceCard;
class L1GctJetLeafCard;
class L1GctEmLeafCard;

class L1GctWheelJetFpga;
class L1GctWheelEnergyFpga;
class L1GctJetFinalStage;
class L1GctGlobalEnergyAlgos;
class L1GctElectronFinalSort;
class L1GctJetEtCalibrationLut;
class L1GctJetCounterLut;


class L1GlobalCaloTrigger {
public:
  /// Number of source cards
  static const int N_SOURCE_CARDS;
  /// Number of Leaf Cards configured for jet processing
  static const int N_JET_LEAF_CARDS;
  /// Number of Leaf Cards configured for EM processing
  static const int N_EM_LEAF_CARDS;
  /// Number of Wheel Cards
  static const int N_WHEEL_CARDS;
  
  /// Number of jet counter per wheel
  static const unsigned int N_JET_COUNTERS_PER_WHEEL;

  /// construct the GCT
  L1GlobalCaloTrigger(bool useFile=false);
  
  /// dismantle the GCT
  ~L1GlobalCaloTrigger();
  
  /// load files into Source Cards
  void openSourceCardFiles(std::string fileBase);
  
  /// Reset internal buffers
  void reset();
  
  /// process an event
  void process();
  
  /// set a jet region at the input to be processed
  void setRegion(L1GctRegion region);

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

  /// setup look-up tables
  void setupLuts();
  void setupJetEtCalibrationLut();
  void setupJetCounterLuts();

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
  
  /// iso electron final stage sorter
  L1GctElectronFinalSort* theIsoEmFinalStage;

  /// non-iso electron final stage sorter
  L1GctElectronFinalSort* theNonIsoEmFinalStage;

  /// Jet Et calibraion LUT
  L1GctJetEtCalibrationLut* m_jetEtCalLut;
  
  /// Jet Counter LUT (Minus Wheel)
  std::vector<L1GctJetCounterLut*> m_minusWheelJetCounterLuts;

  /// Jet Counter LUT (Plus Wheel)
  std::vector<L1GctJetCounterLut*> m_plusWheelJetCounterLuts;

};

#endif /*L1GLOBALCALOTRIGGER_H_*/
