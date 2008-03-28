#ifndef L1GLOBALCALOTRIGGER_H_
#define L1GLOBALCALOTRIGGER_H_

#include "DataFormats/L1CaloTrigger/interface/L1CaloRegion.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloEmCand.h"

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"

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
class L1GctJetCounterSetup;
class L1CaloEtScale;


class L1GlobalCaloTrigger {
public:
  /// Number of Leaf Cards configured for jet processing
  static const int N_JET_LEAF_CARDS;
  /// Number of Leaf Cards configured for EM processing
  static const int N_EM_LEAF_CARDS;
  /// Number of Wheel Cards
  static const int N_WHEEL_CARDS;
  
  /// typedefs for energy values in fixed numbers of bits
  typedef L1GctUnsignedInt< L1GctEtTotal::kEtTotalNBits   > etTotalType;
  typedef L1GctUnsignedInt<   L1GctEtHad::kEtHadNBits     > etHadType;
  typedef L1GctUnsignedInt<  L1GctEtMiss::kEtMissNBits    > etMissType;
  typedef L1GctUnsignedInt<  L1GctEtMiss::kEtMissPhiNBits > etMissPhiType;

  /// construct the GCT
  L1GlobalCaloTrigger(const L1GctJetLeafCard::jetFinderType jfType = L1GctJetLeafCard::tdrJetFinder);

  /// dismantle the GCT
  ~L1GlobalCaloTrigger();
  
  /// Reset internal buffers
  void reset();
  
  /// process an event
  void process();

  ///=================================================================================================
  /// Configuration options for the GCT
  ///
  /// Setup the jet finder parameters
  void setJetFinderParams(const L1GctJetFinderParams* jfpars);

  /// setup the Jet Calibration Lut
  void setJetEtCalibrationLut(const L1GctJetEtCalibrationLut* lut);

  /// setup Jet Counter LUTs
  void setupJetCounterLuts(const L1GctJetCounterSetup* jcPosPars,
                           const L1GctJetCounterSetup* jcNegPars);

  ///=================================================================================================
  /// Multiple bunch operation
  ///
  /// set parameters for multiple bunch operation
  /// process crossings from (firstBx) to (lastBx) 
  void setBxRange(const int firstBx, const int lastBx);
  /// process crossings from (-numOfBx) to (numOfBx) 
  void setBxRangeSymmetric(const int numOfBx);
  /// process all crossings present in the input (and only those crossings)
  void setBxRangeAutomatic();

  /// accessor methods
  int getFirstBx() const { return m_bxStart; }
  int getLastBx()  const { return (m_bxStart + m_numOfBx - 1); }
  int getTotalBx() const { return m_numOfBx; }

  ///=================================================================================================
  /// Input data set methods
  ///
  /// Use the following two methods for full emulator operation 
  /// set jet regions from the RCT at the input to be processed
  void fillRegions(const std::vector<L1CaloRegion>& rgn);

  /// set electrons from the RCT at the input to be processed
  void fillEmCands(const std::vector<L1CaloEmCand>& rgn);

  /// Other methods for debugging
  /// set a jet region at the input to be processed
  void setRegion(const L1CaloRegion& region);

  /// construct a jet region and set it at the input to be processed
  void setRegion(const unsigned et, const unsigned ieta, const unsigned iphi, 
                 const bool overFlow=false, const bool fineGrain=true);

  /// set an isolated EM candidate to be processed
  void setIsoEm(const L1CaloEmCand& em);

  /// set a non-isolated EM candidate to be processed
  void setNonIsoEm(const L1CaloEmCand& em);

  ///=================================================================================================
  /// Output data get methods
  ///
  /// iso electron outputs to GT
  L1GctEmCandCollection getIsoElectrons() const;
  
  /// non-iso electron outputs to GT
  L1GctEmCandCollection getNonIsoElectrons() const;
  
  /// central jet outputs to GT
  L1GctJetCandCollection getCentralJets() const;
  
  /// forward jet outputs to GT
  L1GctJetCandCollection getForwardJets() const;
  
  /// tau jet outputs to GT
  L1GctJetCandCollection getTauJets() const;
  
  /// Total Et output to GT
  L1GctEtTotalCollection getEtSumCollection() const;
  
  /// Total hadronic Et output to GT
  L1GctEtHadCollection   getEtHadCollection() const;

  /// Etmiss output to GT
  L1GctEtMissCollection  getEtMissCollection() const;

  // Jet Count output to GT
  L1GctJetCountsCollection getJetCountsCollection() const;

  ///=================================================================================================
  /// Access to GCT component processors
  ///
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

  ///=================================================================================================
  /// Print method
  ///
  /// print setup info
  void print();
  
 private:
  
  /// Steps in the processing treating input arriving over several bunch crossings
  /// Sort the input data by bunch crossing number
  void sortInputData();
  /// Setup bunch crossing range (depending on input data)
  void bxSetup();
  /// Partial reset for a new bunch crossing
  void bxReset(const int bx);
  /// Fill input data for a new bunch crossing
  void fillEmCands(std::vector<L1CaloEmCand>::iterator& emc, const int bx);
  void fillRegions(std::vector<L1CaloRegion>::iterator& rgn, const int bx);
  /// Process a new bunch crossing
  void bxProcess(const int bx);

  /// instantiate the hardware & algo objects and wire up the system
  void build(L1GctJetLeafCard::jetFinderType jfType);

  /// check we have done all the setup
  bool setupOk() { return (m_jetFinderParams != 0)
                           && (m_jetEtCalLut != 0); }

  /// ordering of the electron sorters to give the correct
  /// priority to the candidates in the final sort 
  unsigned sorterNo(const L1CaloEmCand& em) const;

  /// to process multiple bunch crossings, we need
  /// to select and order input candidates and regions
  /// by beam crossing number
  template <class T>
  struct CompareBx {
    bool operator()(const T& i, const T& j) const {
      return i.bx() < j.bx();
    }
  };
  CompareBx<L1CaloEmCand> emcBxComparator;
  CompareBx<L1CaloRegion> rgnBxComparator;

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

  /// Multiple bunch crossing operation
  bool m_bxRangeAuto;
  int m_bxStart;
  int m_numOfBx;

  /// Local copies of input data, sorted by bunch crossing
  /// then sent to the processors one bunch crossing at a time
  std::vector<L1CaloEmCand> m_allInputEmCands;
  std::vector<L1CaloRegion> m_allInputRegions;
};

#endif /*L1GLOBALCALOTRIGGER_H_*/
