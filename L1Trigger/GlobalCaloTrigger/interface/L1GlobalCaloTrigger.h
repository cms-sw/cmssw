#ifndef L1GLOBALCALOTRIGGER_H_
#define L1GLOBALCALOTRIGGER_H_

#include "DataFormats/L1CaloTrigger/interface/L1CaloRegion.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloEmCand.h"

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEmCand.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCand.h"

//#include "L1Trigger/GlobalCaloTrigger/src/L1GctTwosComplement.h"
#include "L1Trigger/GlobalCaloTrigger/src/L1GctUnsignedInt.h"
#include "L1Trigger/GlobalCaloTrigger/src/L1GctJetCount.h"

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


class L1GctJetLeafCard;
class L1GctJetFinderBase;
class L1GctEmLeafCard;
class L1GctElectronSorter;

class L1GctWheelJetFpga;
class L1GctWheelEnergyFpga;
class L1GctJetFinalStage;
class L1GctGlobalEnergyAlgos;
class L1GctElectronFinalSort;
class L1GctJetFinderParams;
class L1GctJetEtCalibrationLut;
class L1CaloEtScale;


class L1GlobalCaloTrigger {
public:
  /// Number of Leaf Cards configured for jet processing
  static const int N_JET_LEAF_CARDS;
  /// Number of Leaf Cards configured for EM processing
  static const int N_EM_LEAF_CARDS;
  /// Number of Wheel Cards
  static const int N_WHEEL_CARDS;
  
  /// Number of jet counter per wheel
  static const unsigned int N_JET_COUNTERS;

  /// construct the GCT
  L1GlobalCaloTrigger(L1GctJetLeafCard::jetFinderType jfType = L1GctJetLeafCard::tdrJetFinder);

  /// dismantle the GCT
  ~L1GlobalCaloTrigger();
  
  /// Reset internal buffers
  void reset();
  
  /// process an event
  void process();

  /// Setup the jet finder parameters
  void setJetFinderParams(const L1GctJetFinderParams* jfpars);

  /// setup the Jet Calibration Lut
  void setJetEtCalibrationLut(const L1GctJetEtCalibrationLut* lut);

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
  std::vector<L1GctJetCand> getCentralJets() const;
  
  /// forward jet outputs to GT
  std::vector<L1GctJetCand> getForwardJets() const;
  
  /// tau jet outputs to GT
  std::vector<L1GctJetCand> getTauJets() const;
  
  /// Total Et output to GT
  L1GctUnsignedInt<12> getEtSum() const;
  
  /// Total hadronic Et output to GT
  L1GctUnsignedInt<12> getEtHad() const;

  /// Etmiss output to GT
  L1GctUnsignedInt<12> getEtMiss() const;
  
  /// Etmiss phi output to GT
  L1GctUnsignedInt<7> getEtMissPhi() const;

  // Jet Count output to GT
  L1GctJetCount<5> getJetCount(unsigned jcnum) const;
  std::vector<unsigned> getJetCountValues() const;

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
  const L1GctJetEtCalibrationLut* getJetEtCalibLut() const { return m_jetEtCalLut; }

  /// print setup info
  void print();
  
 private:
  
  /// instantiate the hardware & algo objects and wire up the system
  void build(L1GctJetLeafCard::jetFinderType jfType);

  /// setup Jet Counter LUTs
  void setupJetCounterLuts();

  /// check we have done all the setup
  bool setupOk() { return (m_jetFinderParams != 0)
                           && (m_jetEtCalLut != 0); }
 private:
  
  /// pointers to the Jet Leaf cards
  std::vector<L1GctJetLeafCard*> theJetLeafCards;
  
  /// pointers to the Jet Finders
  std::vector<L1GctJetFinderBase*> theJetFinders;
  
  /// pointers to the EM Leaf cards
  std::vector<L1GctEmLeafCard*> theEmLeafCards;
  
  /// pointers to the electron sorters
  std::vector<L1GctElectronSorter*> theIsoElectronSorters;
  std::vector<L1GctElectronSorter*> theNonIsoElectronSorters;
  
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

  /// Jetfinder parameters
  const L1GctJetFinderParams* m_jetFinderParams;

  /// Jet Et calibration LUT
  const L1GctJetEtCalibrationLut* m_jetEtCalLut;

};

#endif /*L1GLOBALCALOTRIGGER_H_*/
