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
 * $Date$
 * $Revision$
 *
 */

// system include files
#include <string>
#include <vector>
#include <map>

#include <boost/cstdint.hpp>

// user include files
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"

// forward declarations

// class declaration
class L1GlobalTriggerReadoutSetup
{

public:
    L1GlobalTriggerReadoutSetup();
    virtual ~L1GlobalTriggerReadoutSetup();

public:

    static const unsigned int NumberPhysTriggers = 128;
    static const unsigned int NumberPhysTriggersExtended = 64; // in addition to 128
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
    static const unsigned int MuonEtaBits = 6; // MSB: sign (0+/1-), 5 bits: value

    // e-gamma, jet objects have 16 bits
    static const unsigned int NumberCaloBits = 16;
    static const unsigned int CaloEtaBits = 4; // MSB: sign (0+/1-), 3 bits: value

    // missing Et has 32 bits
    static const unsigned int NumberMissingEtBits = 32;

    // twelve jet counts, encoded in five bits per count; six jets per 32-bit word
    // code jet count = 31 indicate overflow condition
    static const unsigned int NumberJetCountsBits = 32;
    static const unsigned int NumberJetCountsWords = 2;
    static const unsigned int NumberCountBits = 5;

public:

    // hardware-related stuff

    /// board type and index of the board
    struct GtBoard
    {
        GtBoardType boardType;
        int boardIndex;
        
        boost::uint16_t boardId() const;
        
        bool operator< (const GtBoard&) const; 
    };

    /// number of PSB boards in GT
    static const int NumberPsbBoards = 7;

    /// L1 GT DAQ record map
    ///    gives the position of each block in the GT DAQ readout record
    std::map<int, GtBoard> GtDaqRecordMap;

    /// L1 GT EVM record map
    ///    gives the position of each block in the GT EVM readout record
    std::map<int, GtBoard> GtEvmRecordMap;

    /// L1 GT active boards map
    ///    gives the bit of each GT board in the GTFE ACTIVE_BOARDS
    ///    for the GT DAQ readout record
    std::map<GtBoard, int> GtDaqActiveBoardsMap;

    /// L1 GT active boards map
    ///    gives the bit of each GT board in the GTFE ACTIVE_BOARDS
    ///    for the GT EVM readout record
    std::map<GtBoard, int> GtEvmActiveBoardsMap;

    /// L1 GT board - slot map
    ///    gives the slot of each GT board (part of Board_Id)
    std::map<GtBoard, int> GtBoardSlotMap;

    /// L1 GT board name in hw record map
    ///    gives the bits written for each GT board in the Board_Id
    std::map<GtBoardType, int> GtBoardHexNameMap;

public:

    /// L1 GT calo input map
    ///    gives the mapping of calorimeter objects to GT calorimeter input
    /// GT calorimeter input will be mapped to PSBs later
    std::map<int, L1GtCaloQuad> GtCaloObjectInputMap;


    /// L1 GT calo input to PSB map
    ///    gives the mapping of GT calorimeter input to GT PSBs
    std::map<int, int> GtCaloInputToPsbMap;


};

#endif /*L1GlobalTrigger_L1GlobalTriggerReadoutSetup_h*/
