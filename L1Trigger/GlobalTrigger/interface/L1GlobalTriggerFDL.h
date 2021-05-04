#ifndef GlobalTrigger_L1GlobalTriggerFDL_h
#define GlobalTrigger_L1GlobalTriggerFDL_h
/**
 * \class L1GlobalTriggerFDL
 *
 *
 * Description: Final Decision Logic board.
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
#include <vector>

// user include files
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"

#include "FWCore/Framework/interface/Event.h"

#include "CondFormats/L1TObjects/interface/L1GtBoard.h"
#include "CondFormats/L1TObjects/interface/L1GtFwd.h"
#include <cstdint>

// forward declarations
class L1GlobalTriggerReadoutRecord;
class L1GlobalTriggerEvmReadoutRecord;

class L1GtFdlWord;
class L1GlobalTriggerGTL;
class L1GlobalTriggerPSB;

// class declaration
class L1GlobalTriggerFDL {
public:
  /// constructor
  L1GlobalTriggerFDL();

  /// destructor
  virtual ~L1GlobalTriggerFDL();

  /// run the FDL
  void run(edm::Event &iEvent,
           const std::vector<int> &prescaleFactorsAlgoTrig,
           const std::vector<int> &prescaleFactorsTechTrig,
           const std::vector<unsigned int> &triggerMaskAlgoTrig,
           const std::vector<unsigned int> &triggerMaskTechTrig,
           const std::vector<unsigned int> &triggerMaskVetoAlgoTrig,
           const std::vector<unsigned int> &triggerMaskVetoTechTrig,
           const std::vector<L1GtBoard> &boardMaps,
           const int totalBxInEvent,
           const int iBxInEvent,
           const unsigned int numberPhysTriggers,
           const unsigned int numberTechnicalTriggers,
           const unsigned int numberDaqPartitions,
           const L1GlobalTriggerGTL *ptrGTL,
           const L1GlobalTriggerPSB *ptrPSB,
           const int pfAlgoSetIndex,
           const int pfTechSetIndex,
           const bool algorithmTriggersUnprescaled,
           const bool algorithmTriggersUnmasked,
           const bool technicalTriggersUnprescaled,
           const bool technicalTriggersUnmasked,
           const bool technicalTriggersVetoUnmasked);

  /// fill the FDL block in the L1 GT DAQ record for iBxInEvent
  void fillDaqFdlBlock(const int iBxInEvent,
                       const uint16_t &activeBoardsGtDaq,
                       const int recordLength0,
                       const int recordLength1,
                       const unsigned int altNrBxBoardDaq,
                       const std::vector<L1GtBoard> &boardMaps,
                       L1GlobalTriggerReadoutRecord *gtDaqReadoutRecord);

  /// fill the FDL block in the L1 GT EVM record for iBxInEvent
  void fillEvmFdlBlock(const int iBxInEvent,
                       const uint16_t &activeBoardsGtEvm,
                       const int recordLength0,
                       const int recordLength1,
                       const unsigned int altNrBxBoardEvm,
                       const std::vector<L1GtBoard> &boardMaps,
                       L1GlobalTriggerEvmReadoutRecord *gtEvmReadoutRecord);

  /// clear FDL
  void reset();

  /// return the GtFdlWord
  inline L1GtFdlWord *gtFdlWord() const { return m_gtFdlWord; }

public:
  inline void setVerbosity(const int verbosity) { m_verbosity = verbosity; }

private:
  L1GtFdlWord *m_gtFdlWord;

  /// prescale counters: NumberPhysTriggers counters per bunch cross in event
  std::vector<std::vector<int>> m_prescaleCounterAlgoTrig;

  /// prescale counters: technical trigger counters per bunch cross in event
  std::vector<std::vector<int>> m_prescaleCounterTechTrig;

  /// logical switches for
  ///    the first event
  ///    the first event in the luminosity segment
  ///    and the first event in the run
  bool m_firstEv;
  bool m_firstEvLumiSegment;
  bool m_firstEvRun;

private:
  /// verbosity level
  int m_verbosity;
  bool m_isDebugEnabled;
};

#endif
