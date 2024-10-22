#ifndef GlobalTrigger_L1GlobalTriggerPSB_h
#define GlobalTrigger_L1GlobalTriggerPSB_h

/**
 * \class L1GlobalTriggerPSB
 *
 *
 * Description: Pipelined Synchronising Buffer.
 *
 * Implementation:
 *    GT PSB receives data from
 *      - Global Calorimeter Trigger
 *      - Technical Trigger
 *
 * \author: M. Fierro            - HEPHY Vienna - ORCA version
 * \author: Vasile Mihai Ghete   - HEPHY Vienna - CMSSW version
 *
 *
 */

// system include files
#include <vector>

// user include files
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GtTechnicalTriggerRecord.h"

#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "CondFormats/L1TObjects/interface/L1GtBoard.h"
#include "CondFormats/L1TObjects/interface/L1GtBoardMaps.h"
#include "CondFormats/L1TObjects/interface/L1GtFwd.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include <cstdint>

// forward declarations
class L1GctCand;

class L1GctEmCand;
class L1GctJetCand;

class L1GctEtMiss;
class L1GctEtTotal;
class L1GctEtHad;
class L1GctHtMiss;

class L1GctJetCounts;
class L1GctHFBitCounts;
class L1GctHFRingEtSums;

class L1GlobalTriggerReadoutRecord;

// class declaration
class L1GlobalTriggerPSB {
public:
  // constructor
  L1GlobalTriggerPSB(const edm::InputTag &caloTag,
                     const std::vector<edm::InputTag> &vecTag,
                     edm::ConsumesCollector &&iC);

  // destructor
  virtual ~L1GlobalTriggerPSB();

public:
  /// initialize the class (mainly reserve)
  void init(const int nrL1NoIsoEG,
            const int nrL1IsoEG,
            const int nrL1CenJet,
            const int nrL1ForJet,
            const int nrL1TauJet,
            const int numberTechnicalTriggers);

  /// receive Global Calorimeter Trigger objects
  void receiveGctObjectData(edm::Event &iEvent,
                            const edm::InputTag &caloGctInputTag,
                            const int iBxInEvent,
                            const bool receiveNoIsoEG,
                            const int nrL1NoIsoEG,
                            const bool receiveIsoEG,
                            const int nrL1IsoEG,
                            const bool receiveCenJet,
                            const int nrL1CenJet,
                            const bool receiveForJet,
                            const int nrL1ForJet,
                            const bool receiveTauJet,
                            const int nrL1TauJet,
                            const bool receiveETM,
                            const bool receiveETT,
                            const bool receiveHTT,
                            const bool receiveHTM,
                            const bool receiveJetCounts,
                            const bool receiveHfBitCounts,
                            const bool receiveHfRingEtSums);

  /// receive CASTOR objects
  void receiveCastorData(edm::Event &iEvent,
                         const edm::InputTag &castorInputTag,
                         const int iBxInEvent,
                         const bool receiveCastor,
                         const bool readFromPsb);

  /// receive BPTX objects
  void receiveBptxData(edm::Event &iEvent,
                       const edm::InputTag &bptxInputTag,
                       const int iBxInEvent,
                       const bool receiveBptx,
                       const bool readFromPsb);

  /// receive External objects
  void receiveExternalData(edm::Event &iEvent,
                           const std::vector<edm::InputTag> &externalInputTags,
                           const int iBxInEvent,
                           const bool receiveExternal,
                           const bool readFromPsb);

  /// receive technical trigger
  void receiveTechnicalTriggers(edm::Event &iEvent,
                                const std::vector<edm::InputTag> &technicalTriggersInputTags,
                                const int iBxInEvent,
                                const bool receiveTechTr,
                                const int nrL1TechTr);

  /// fill the content of active PSB boards
  void fillPsbBlock(edm::Event &iEvent,
                    const uint16_t &activeBoardsGtDaq,
                    const int recordLength0,
                    const int recordLength1,
                    const unsigned int altNrBxBoardDaq,
                    const std::vector<L1GtBoard> &boardMaps,
                    const int iBxInEvent,
                    L1GlobalTriggerReadoutRecord *gtDaqReadoutRecord);

  /// clear PSB
  void reset();

  /// print Global Calorimeter Trigger data
  void printGctObjectData(const int iBxInEvent) const;

  /// pointer to NoIsoEG data list
  inline const std::vector<const L1GctCand *> *getCandL1NoIsoEG() const { return m_candL1NoIsoEG; }

  /// pointer to IsoEG data list
  inline const std::vector<const L1GctCand *> *getCandL1IsoEG() const { return m_candL1IsoEG; }

  /// pointer to CenJet data list
  inline const std::vector<const L1GctCand *> *getCandL1CenJet() const { return m_candL1CenJet; }

  /// pointer to ForJet data list
  inline const std::vector<const L1GctCand *> *getCandL1ForJet() const { return m_candL1ForJet; }

  /// pointer to TauJet data list
  inline const std::vector<const L1GctCand *> *getCandL1TauJet() const { return m_candL1TauJet; }

  /// pointer to ETM data list
  inline const L1GctEtMiss *getCandL1ETM() const { return m_candETM; }

  /// pointer to ETT data list
  inline const L1GctEtTotal *getCandL1ETT() const { return m_candETT; }

  /// pointer to HTT data list
  inline const L1GctEtHad *getCandL1HTT() const { return m_candHTT; }

  /// pointer to HTM data list
  inline const L1GctHtMiss *getCandL1HTM() const { return m_candHTM; }

  /// pointer to JetCounts data list
  inline const L1GctJetCounts *getCandL1JetCounts() const { return m_candJetCounts; }

  /// pointer to HfBitCounts data list
  inline const L1GctHFBitCounts *getCandL1HfBitCounts() const { return m_candHfBitCounts; }

  /// pointer to HfRingEtSums data list
  inline const L1GctHFRingEtSums *getCandL1HfRingEtSums() const { return m_candHfRingEtSums; }

  /// pointer to technical trigger bits
  inline const std::vector<bool> *getGtTechnicalTriggers() const { return &m_gtTechnicalTriggers; }

public:
  inline void setVerbosity(const int verbosity) { m_verbosity = verbosity; }

private:
  std::vector<const L1GctCand *> *m_candL1NoIsoEG;
  std::vector<const L1GctCand *> *m_candL1IsoEG;
  std::vector<const L1GctCand *> *m_candL1CenJet;
  std::vector<const L1GctCand *> *m_candL1ForJet;
  std::vector<const L1GctCand *> *m_candL1TauJet;

  const L1GctEtMiss *m_candETM;
  const L1GctEtTotal *m_candETT;
  const L1GctEtHad *m_candHTT;
  const L1GctHtMiss *m_candHTM;

  const L1GctJetCounts *m_candJetCounts;

  const L1GctHFBitCounts *m_candHfBitCounts;
  const L1GctHFRingEtSums *m_candHfRingEtSums;

  /// technical trigger bits
  std::vector<bool> m_gtTechnicalTriggers;

private:
  /// verbosity level
  int m_verbosity;
  bool m_isDebugEnabled;
};

#endif
