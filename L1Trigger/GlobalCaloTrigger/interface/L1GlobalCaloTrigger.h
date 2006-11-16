#ifndef L1GLOBALCALOTRIGGER_H_
#define L1GLOBALCALOTRIGGER_H_

#include "DataFormats/L1CaloTrigger/interface/L1CaloRegion.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloEmCand.h"

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEmCand.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJet.h"
//#include "L1Trigger/GlobalCaloTrigger/src/L1GctTwosComplement.h"
#include "L1Trigger/GlobalCaloTrigger/src/L1GctUnsignedInt.h"
#include "L1Trigger/GlobalCaloTrigger/src/L1GctJetCounts.h"

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetLeafCard.h"

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
class L1CaloEtScale;


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
  L1GlobalCaloTrigger(bool useFile=false,
		      L1GctJetLeafCard::jetFinderType jfType = L1GctJetLeafCard::tdrJetFinder,
		      std::string jetEtLutFile="data/defaultJetEtCalibrationLut.dat");

  /// dismantle the GCT
  ~L1GlobalCaloTrigger();
  
  /// load files into Source Cards
  void openSourceCardFiles(std::string fileBase);
  
  /// Reset internal buffers
  void reset();
  
  /// process an event
  void process();

  /// set a jet region at the input to be processed
  void setRegion(L1CaloRegion region);

  /// construct a jet region and set it at the input to be processed
  void setRegion(unsigned et, unsigned ieta, unsigned iphi, bool overFlow=false, bool fineGrain=true);

  /// set an isolated EM candidate to be processed
  void setIsoEm(L1CaloEmCand em);

  /// set a non-isolated EM candidate to be processed
  void setNonIsoEm(L1CaloEmCand em);

  /// set jet regions from the RCT at the input to be processed
  void fillRegions(std::vector<L1CaloRegion> rgn);

  /// set electrons from the RCT at the input to be processed
  void fillEmCands(std::vector<L1CaloEmCand> rgn);

  /// iso electron outputs to GT
  std::vector<L1GctEmCand> getIsoElectrons() const;
  
  /// non-iso electron outputs to GT
  std::vector<L1GctEmCand> getNonIsoElectrons() const;
  
  /// central jet outputs to GT
  std::vector<L1GctJet> getCentralJets() const;
  
  /// forward jet outputs to GT
  std::vector<L1GctJet> getForwardJets() const;
  
  /// tau jet outputs to GT
  std::vector<L1GctJet> getTauJets() const;
  
  /// Total Et output to GT
  L1GctScalarEtVal getEtSum() const;
  
  /// Total hadronic Et output to GT
  L1GctScalarEtVal getEtHad() const;

  /// Etmiss output to GT
  L1GctScalarEtVal getEtMiss() const;
  
  /// Etmiss phi output to GT
  L1GctEtAngleBin getEtMissPhi() const;

  // Jet Count output to GT
  L1GctJcFinalType getJetCount(unsigned jcnum) const;

  /// get the Source cards
  std::vector<L1GctSourceCard*> getSourceCards() const { return theSourceCards; }
  
  /// get the Jet Leaf cards
  std::vector<L1GctJetLeafCard*> getJetLeafCards() const { return theJetLeafCards; }
  
  /// get the Jet Leaf cards
  std::vector<L1GctEmLeafCard*> getEmLeafCards() const { return theEmLeafCards; }
  
  /// get the Wheel Jet FPGAs
  std::vector<L1GctWheelJetFpga*> getWheelJetFpgas() const { return theWheelJetFpgas; }
  
  /// get the Wheel Energy Fpgas
  std::vector<L1GctWheelEnergyFpga*> getWheelEnergyFpgas() const { return theWheelEnergyFpgas; }
  
  /// get the jet final stage
  L1GctJetFinalStage* getJetFinalStage() const { return theJetFinalStage; }
  
  /// get the energy final stage
  L1GctGlobalEnergyAlgos* getEnergyFinalStage() const { return theEnergyFinalStage; }
  
  /// get the electron final stage sorters
  L1GctElectronFinalSort* getIsoEmFinalStage() const { return theIsoEmFinalStage; }
  L1GctElectronFinalSort* getNonIsoEmFinalStage() const { return theNonIsoEmFinalStage; }

  /// get the Jet Et calibration LUT
  L1GctJetEtCalibrationLut* getJetEtCalibLut() const { return m_jetEtCalLut; }

  /// print setup info
  void print();
  
 private:
  
  /// instantiate the hardware & algo objects and wire up the system
  void build(L1GctJetLeafCard::jetFinderType jfType);

  /// setup Jet Counter LUTs
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

  /// Jet Et calibration LUT
  L1GctJetEtCalibrationLut* m_jetEtCalLut;

  /// default jet rank scale
  L1CaloEtScale* m_defaultJetEtScale;
  
};

#endif /*L1GLOBALCALOTRIGGER_H_*/
