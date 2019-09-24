#ifndef GlobalTrigger_L1GlobalTrigger_h
#define GlobalTrigger_L1GlobalTrigger_h

/**
 * \class L1GlobalTrigger
 *
 *
 * Description: L1 Global Trigger producer.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 *
 *
 * The CMSSW implementation of the L1 Global Trigger emulator
 * uses concepts and code from the ORCA L1 Global Trigger simulation,
 * authors: N. Neumeister, M. Fierro, M. Eder  - HEPHY Vienna.
 *
 */

// system include files
#include <string>
#include <vector>

// user include files

#include "CondFormats/L1TObjects/interface/L1GtBoard.h"
#include "CondFormats/L1TObjects/interface/L1GtFwd.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include <cstdint>

// forward classes
class L1GlobalTriggerPSB;
class L1GlobalTriggerGTL;
class L1GlobalTriggerFDL;

class L1GtStableParameters;
class L1GtParameters;
class L1GtBoardMaps;

class L1GtPrescaleFactors;
class L1GtTriggerMask;

// class declaration
class L1GlobalTrigger : public edm::stream::EDProducer<> {
public:
  explicit L1GlobalTrigger(const edm::ParameterSet &);
  ~L1GlobalTrigger() override;

  void produce(edm::Event &, const edm::EventSetup &) override;

  // return pointer to PSB
  inline const L1GlobalTriggerPSB *gtPSB() const { return m_gtPSB; }

  // return pointer to GTL
  inline const L1GlobalTriggerGTL *gtGTL() const { return m_gtGTL; }

  // return pointer to FDL
  inline const L1GlobalTriggerFDL *gtFDL() const { return m_gtFDL; }

private:
  /// cached stuff

  /// stable parameters
  const L1GtStableParameters *m_l1GtStablePar;
  unsigned long long m_l1GtStableParCacheID;

  /// number of physics triggers
  unsigned int m_numberPhysTriggers;

  /// number of technical triggers
  unsigned int m_numberTechnicalTriggers;

  /// number of DAQ partitions
  unsigned int m_numberDaqPartitions;

  /// number of objects of each type
  ///    { Mu, NoIsoEG, IsoEG, CenJet, ForJet, TauJet, ETM, ETT, HTT, JetCounts
  ///    };
  int m_nrL1Mu;

  int m_nrL1NoIsoEG;
  int m_nrL1IsoEG;

  int m_nrL1CenJet;
  int m_nrL1ForJet;
  int m_nrL1TauJet;

  int m_nrL1JetCounts;

  // ... the rest of the objects are global

  int m_ifMuEtaNumberBits;
  int m_ifCaloEtaNumberBits;

  /// parameters
  const L1GtParameters *m_l1GtPar;
  unsigned long long m_l1GtParCacheID;

  ///    total number of Bx's in the event coming from EventSetup
  int m_totalBxInEvent;

  ///    active boards in L1 GT DAQ record and in L1 GT EVM record
  uint16_t m_activeBoardsGtDaq;
  uint16_t m_activeBoardsGtEvm;

  /// length of BST record (in bytes) from event setup
  unsigned int m_bstLengthBytes;

  /// board maps - cache only the record
  const L1GtBoardMaps *m_l1GtBM;
  unsigned long long m_l1GtBMCacheID;

  /// prescale factors
  const L1GtPrescaleFactors *m_l1GtPfAlgo;
  unsigned long long m_l1GtPfAlgoCacheID;

  const L1GtPrescaleFactors *m_l1GtPfTech;
  unsigned long long m_l1GtPfTechCacheID;

  const std::vector<std::vector<int>> *m_prescaleFactorsAlgoTrig;
  const std::vector<std::vector<int>> *m_prescaleFactorsTechTrig;

  /// trigger masks & veto masks
  const L1GtTriggerMask *m_l1GtTmAlgo;
  unsigned long long m_l1GtTmAlgoCacheID;

  const L1GtTriggerMask *m_l1GtTmTech;
  unsigned long long m_l1GtTmTechCacheID;

  const L1GtTriggerMask *m_l1GtTmVetoAlgo;
  unsigned long long m_l1GtTmVetoAlgoCacheID;

  const L1GtTriggerMask *m_l1GtTmVetoTech;
  unsigned long long m_l1GtTmVetoTechCacheID;

  std::vector<unsigned int> m_triggerMaskAlgoTrig;
  std::vector<unsigned int> m_triggerMaskTechTrig;

  std::vector<unsigned int> m_triggerMaskVetoAlgoTrig;
  std::vector<unsigned int> m_triggerMaskVetoTechTrig;

  L1GlobalTriggerPSB *m_gtPSB;
  L1GlobalTriggerGTL *m_gtGTL;
  L1GlobalTriggerFDL *m_gtFDL;

  /// input tag for muon collection from GMT
  const edm::InputTag m_muGmtInputTag;

  /// input tag for calorimeter collections from GCT
  const edm::InputTag m_caloGctInputTag;

  /// input tag for CASTOR record
  const edm::InputTag m_castorInputTag;

  /// input tag for technical triggers
  const std::vector<edm::InputTag> m_technicalTriggersInputTags;

  /// logical flag to produce the L1 GT DAQ readout record
  const bool m_produceL1GtDaqRecord;

  /// logical flag to produce the L1 GT EVM readout record
  const bool m_produceL1GtEvmRecord;

  /// logical flag to produce the L1 GT object map record
  const bool m_produceL1GtObjectMapRecord;

  /// logical flag to write the PSB content in the  L1 GT DAQ record
  const bool m_writePsbL1GtDaqRecord;

  /// logical flag to read the technical trigger records
  const bool m_readTechnicalTriggerRecords;

  /// number of "bunch crossing in the event" (BxInEvent) to be emulated
  /// symmetric around L1Accept (BxInEvent = 0):
  ///    1 (BxInEvent = 0); 3 (F 0 1) (standard record); 5 (E F 0 1 2) (debug
  ///    record)
  /// even numbers (except 0) "rounded" to the nearest lower odd number
  int m_emulateBxInEvent;

  /// number of BXs in the event corresponding to alternative 0 and 1 in
  /// altNrBxBoard() EmulateBxInEvent >= max(RecordLength[0], RecordLength[1])
  /// negative values: take the numbers from event setup, from L1GtParameters
  const std::vector<int> m_recordLength;

  /// alternative for number of BX per active board in GT DAQ record: 0 or 1
  /// the position is identical with the active board bit
  const unsigned int m_alternativeNrBxBoardDaq;

  /// alternative for number of BX per active board in GT EVM record: 0 or 1
  /// the position is identical with the active board bit
  const unsigned int m_alternativeNrBxBoardEvm;

  /// length of BST record (in bytes) from parameter set
  const int m_psBstLengthBytes;

  /// run algorithm triggers
  ///     if true, unprescaled (all prescale factors 1)
  ///     will overwrite the event setup
  const bool m_algorithmTriggersUnprescaled;

  ///     if true, unmasked - all enabled (all trigger masks set to 0)
  ///     will overwrite the event setup
  const bool m_algorithmTriggersUnmasked;

  /// run technical triggers
  ///     if true, unprescaled (all prescale factors 1)
  ///     will overwrite the event setup
  const bool m_technicalTriggersUnprescaled;

  ///     if true, unmasked - all enabled (all trigger masks set to 0)
  ///     will overwrite the event setup
  const bool m_technicalTriggersUnmasked;

  ///     if true, veto unmasked - all enabled (all trigger veto masks set to 0)
  ///     will overwrite the event setup
  const bool m_technicalTriggersVetoUnmasked;

  /// verbosity level
  const int m_verbosity;
  const bool m_isDebugEnabled;
};

#endif /*GlobalTrigger_L1GlobalTrigger_h*/
