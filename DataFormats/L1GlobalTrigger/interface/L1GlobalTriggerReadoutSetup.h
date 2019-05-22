#ifndef L1GlobalTrigger_L1GlobalTriggerReadoutSetup_h
#define L1GlobalTrigger_L1GlobalTriggerReadoutSetup_h

/**
 * \class L1GlobalTriggerReadoutSetup
 * 
 * 
 * Description: group static constants for GT readout record.  
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 * 
 *
 */

// system include files
#include <string>
#include <vector>
#include <map>

// user include files
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "FWCore/Utilities/interface/typedefs.h"

// forward declarations

// class declaration
class L1GlobalTriggerReadoutSetup {
public:
  L1GlobalTriggerReadoutSetup();
  virtual ~L1GlobalTriggerReadoutSetup();

public:
  static const unsigned int NumberPhysTriggers = 128;
  static const unsigned int NumberPhysTriggersExtended = 64;  // in addition to 128
  static const unsigned int NumberTechnicalTriggers = 64;

  static const unsigned int NumberL1Muons = 4;

  static const unsigned int NumberL1Electrons = 4;
  static const unsigned int NumberL1IsolatedElectrons = 4;

  static const unsigned int NumberL1CentralJets = 4;
  static const unsigned int NumberL1ForwardJets = 4;
  static const unsigned int NumberL1TauJets = 4;

  static const unsigned int NumberL1JetCounts = 12;

public:
  /// GT DAQ record organized in words of WordLength bits
  static const int WordLength = 64;

  /// one unit in the word is UnitLength bits
  static const int UnitLength = 8;

public:
  // muons are represented as 32 bits (actually 26 bits)
  static const unsigned int NumberMuonBits = 32;
  static const unsigned int MuonEtaBits = 6;  // MSB: sign (0+/1-), 5 bits: value

  // e-gamma, jet objects have 16 bits
  static const unsigned int NumberCaloBits = 16;
  static const unsigned int CaloEtaBits = 4;  // MSB: sign (0+/1-), 3 bits: value

  // missing Et has 32 bits
  static const unsigned int NumberMissingEtBits = 32;

  // twelve jet counts, encoded in five bits per count; six jets per 32-bit word
  // code jet count = 31 indicate overflow condition
  static const unsigned int NumberJetCountsBits = 32;
  static const unsigned int NumberJetCountsWords = 2;
  static const unsigned int NumberCountBits = 5;

  /// number of PSB boards in GT
  static const int NumberPsbBoards = 7;
};

#endif /*L1GlobalTrigger_L1GlobalTriggerReadoutSetup_h*/
