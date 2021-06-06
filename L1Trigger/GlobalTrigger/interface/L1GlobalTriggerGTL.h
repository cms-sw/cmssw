#ifndef GlobalTrigger_L1GlobalTriggerGTL_h
#define GlobalTrigger_L1GlobalTriggerGTL_h

/**
 * \class L1GlobalTriggerGTL
 *
 *
 * Description: Global Trigger Logic board.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author: M. Fierro            - HEPHY Vienna - ORCA version
 * \author: Vasile Mihai Ghete   - HEPHY Vienna - CMSSW version
 *
 *
 */

// system include files
#include <bitset>
#include <vector>

// user include files
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"
#include "CondFormats/DataRecord/interface/L1CaloGeometryRecord.h"
#include "CondFormats/DataRecord/interface/L1MuTriggerScalesRcd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "L1Trigger/GlobalTrigger/interface/L1GtAlgorithmEvaluation.h"

#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTCand.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/EventSetup.h"

// forward declarations
class L1GlobalTriggerPSB;
class L1GtTriggerMenu;
class L1CaloGeometry;
class L1MuTriggerScales;
class L1GtEtaPhiConversions;

// class declaration
class L1GlobalTriggerGTL {
public:
  // constructors
  L1GlobalTriggerGTL(const edm::InputTag &mutag, edm::ConsumesCollector &&iC);

  // destructor
  virtual ~L1GlobalTriggerGTL();

public:
  /// receive data from Global Muon Trigger
  void receiveGmtObjectData(
      edm::Event &, const edm::InputTag &, const int iBxInEvent, const bool receiveMu, const int nrL1Mu);

  /// initialize the class (mainly reserve)
  void init(const int nrL1Mu, const int numberPhysTriggers);

  /// run the GTL
  void run(edm::Event &iEvent,
           const edm::EventSetup &evSetup,
           const L1GlobalTriggerPSB *ptrGtPSB,
           const bool produceL1GtObjectMapRecord,
           const int iBxInEvent,
           L1GlobalTriggerObjectMapRecord *gtObjectMapRecord,
           const unsigned int numberPhysTriggers,
           const int nrL1Mu,
           const int nrL1NoIsoEG,
           const int nrL1IsoEG,
           const int nrL1CenJet,
           const int nrL1ForJet,
           const int nrL1TauJet,
           const int nrL1JetCounts,
           const int ifMuEtaNumberBits,
           const int ifCaloEtaNumberBits);

  /// clear GTL
  void reset();

  /// print received Muon dataWord
  void printGmtData(const int iBxInEvent) const;

  /// return decision
  inline const std::bitset<L1GlobalTriggerReadoutSetup::NumberPhysTriggers> &getDecisionWord() const {
    return m_gtlDecisionWord;
  }

  /// return algorithm OR decision
  inline const std::bitset<L1GlobalTriggerReadoutSetup::NumberPhysTriggers> &getAlgorithmOR() const {
    return m_gtlAlgorithmOR;
  }

  /// return global muon trigger candidate
  inline const std::vector<const L1MuGMTCand *> *getCandL1Mu() const { return m_candL1Mu; }

public:
  inline void setVerbosity(const int verbosity) { m_verbosity = verbosity; }

private:
  // cached stuff

  // trigger menu
  const edm::ESGetToken<L1GtTriggerMenu, L1GtTriggerMenuRcd> m_l1GtMenuToken;

  // L1 scales (phi, eta) for Mu, Calo and EnergySum objects
  const edm::ESGetToken<L1CaloGeometry, L1CaloGeometryRecord> m_l1CaloGeometryToken;
  const L1CaloGeometry *m_l1CaloGeometry;
  unsigned long long m_l1CaloGeometryCacheID;

  const edm::ESGetToken<L1MuTriggerScales, L1MuTriggerScalesRcd> m_l1MuTriggerScalesToken;
  const L1MuTriggerScales *m_l1MuTriggerScales;
  unsigned long long m_l1MuTriggerScalesCacheID;

  // conversions for eta and phi
  L1GtEtaPhiConversions *m_gtEtaPhiConversions;

private:
  std::vector<const L1MuGMTCand *> *m_candL1Mu;

  std::bitset<L1GlobalTriggerReadoutSetup::NumberPhysTriggers> m_gtlAlgorithmOR;
  std::bitset<L1GlobalTriggerReadoutSetup::NumberPhysTriggers> m_gtlDecisionWord;

  // cache of maps
  std::vector<L1GtAlgorithmEvaluation::ConditionEvaluationMap> m_conditionResultMaps;

private:
  /// verbosity level
  int m_verbosity;
  bool m_isDebugEnabled;
};

#endif
