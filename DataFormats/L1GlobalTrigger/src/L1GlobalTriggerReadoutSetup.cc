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

// this class header
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"

// system include files
// user include files
//   base class
// forward declarations

// constructor
L1GlobalTriggerReadoutSetup::L1GlobalTriggerReadoutSetup()
{

    // create boards
    GtBoard gtfeBoard = {GTFE, 0};

    GtBoard tcsBoard = {TCS, 0};

    GtBoard timBoard = {TIM, 0};

    GtBoard fdlBoard = {FDL, 0};

    // later, loop over (int iPsb = 0; iPsb < NumberPsbBoards;) to set boardIndex
    GtBoard psbBoard;
    psbBoard.boardType = PSB;

    GtBoard gmtBoard = {GMT, 0};


    //populate the maps

    // L1 GT DAQ record map
    //    gives the position of each block in the GT DAQ readout record

    // header has position 0

    // GTFE position 1
    int iBoard = 1;

    GtDaqRecordMap[iBoard] = gtfeBoard;
    iBoard++;

    GtDaqRecordMap[iBoard] = fdlBoard;
    iBoard++;

    for (int iPsb = 0; iPsb < NumberPsbBoards; ++iPsb) {
        psbBoard.boardIndex = iPsb;
        GtDaqRecordMap[iBoard] = psbBoard;
        iBoard++;
    }

    GtDaqRecordMap[iBoard] = gmtBoard;

    // trailer has latest position

    // L1 GT EVM record map
    //    gives the position of each block in the GT EVM readout record

    // header has position 0

    // GTFE position 1
    int iBoardEvm = 1;

    GtEvmRecordMap[iBoardEvm] = gtfeBoard;
    iBoardEvm++;

    GtEvmRecordMap[iBoardEvm] = tcsBoard;
    iBoardEvm++;

    GtEvmRecordMap[iBoardEvm] = fdlBoard;
    iBoardEvm++;

    // trailer has latest position


    // L1 GT active boards map
    //    gives the bit of each GT board in the GTFE ACTIVE_BOARDS
    //    for the GT DAQ readout record

    int iBit = 0;

    GtDaqActiveBoardsMap[fdlBoard] = iBit;
    iBit++;

    for (int iPsb = 0; iPsb < NumberPsbBoards; ++iPsb) {
        psbBoard.boardIndex = iPsb;
        GtDaqActiveBoardsMap[psbBoard] = iBit;
        iBit++;
    }


    GtDaqActiveBoardsMap[gmtBoard] = iBit;
    iBit++;

    gmtBoard.boardIndex = 1; // spare, not used
    GtDaqActiveBoardsMap[gmtBoard] = iBit;
    iBit++;

    GtDaqActiveBoardsMap[tcsBoard] = iBit;
    iBit++;

    GtDaqActiveBoardsMap[timBoard] = iBit;
    iBit++;


    // L1 GT active boards map
    //    gives the bit of each GT board in the GTFE ACTIVE_BOARDS
    //    for the GT EVM readout record

    int iEvmBit = 0;

    GtEvmActiveBoardsMap[tcsBoard] = iEvmBit;
    iEvmBit++;

    GtEvmActiveBoardsMap[fdlBoard] = iEvmBit;
    iBit++;


    // L1 GT board - slot map
    //    gives the slot of each GT board (part of Board_Id)

    int iSlot = 17;
    GtBoardSlotMap[gtfeBoard] = iSlot;

    iSlot = 10;
    GtBoardSlotMap[fdlBoard] = iSlot;

    psbBoard.boardIndex = 0;
    iSlot = 9;
    GtBoardSlotMap[psbBoard] = iSlot;

    psbBoard.boardIndex = 1;
    iSlot = 13;
    GtBoardSlotMap[psbBoard] = iSlot;
    psbBoard.boardIndex = 2;
    iSlot = 14;
    GtBoardSlotMap[psbBoard] = iSlot;
    psbBoard.boardIndex = 3;
    iSlot = 15;
    GtBoardSlotMap[psbBoard] = iSlot;

    psbBoard.boardIndex = 4;
    iSlot = 19;
    GtBoardSlotMap[psbBoard] = iSlot;
    psbBoard.boardIndex = 5;
    iSlot = 20;
    GtBoardSlotMap[psbBoard] = iSlot;
    psbBoard.boardIndex = 6;
    iSlot = 21;
    GtBoardSlotMap[psbBoard] = iSlot;

    gmtBoard.boardIndex = 0;
    iSlot = 18;
    GtBoardSlotMap[gmtBoard] = iSlot;

    gmtBoard.boardIndex = 1;
    iSlot = 18;
    GtBoardSlotMap[gmtBoard] = iSlot;

    iSlot = 7;
    GtBoardSlotMap[tcsBoard] = iSlot;

    iSlot = 16;
    GtBoardSlotMap[timBoard] = iSlot;


    // L1 GT board name in hw record map
    //    gives the bits written for each GT board in the Board_Id

    GtBoardHexNameMap[GTFE] = 0x00; // not used, as only 8 bits are available
    GtBoardHexNameMap[FDL]  = 0xfd;
    GtBoardHexNameMap[PSB]  = 0xbb;
    GtBoardHexNameMap[GMT]  = 0xdd; // TODO GMT_spare has 0xde FIXME when board available
    GtBoardHexNameMap[TCS]  = 0xcc; // TODO TCS_M     has 0xcd FIXME when board available
    GtBoardHexNameMap[TIM]  = 0xad;

    // L1 GT calo input map
    //    gives the mapping of calorimeter objects to GT calorimeter input

    GtCaloObjectInputMap[1] = IsoEGQ;
    GtCaloObjectInputMap[2] = NoIsoEGQ;

    GtCaloObjectInputMap[3] = CenJetQ;
    GtCaloObjectInputMap[4] = ForJetQ;
    GtCaloObjectInputMap[5] = TauJetQ;

    GtCaloObjectInputMap[6] = ESumsQ;
    GtCaloObjectInputMap[7] = JetCountsQ;

    // L1 GT calo input to PSB map
    //    gives the mapping of GT calorimeter input to GT PSBs
    //    4 infinicables per PSB
    GtCaloInputToPsbMap[1] = 1;
    GtCaloInputToPsbMap[2] = 1;
    GtCaloInputToPsbMap[3] = 1;
    GtCaloInputToPsbMap[4] = 1;

    GtCaloInputToPsbMap[5] = 2;
    GtCaloInputToPsbMap[6] = 2;
    GtCaloInputToPsbMap[7] = 2;
    GtCaloInputToPsbMap[8] = 2;

    // PSB_3 can be used only 50% due to missing connections to GTL
    GtCaloInputToPsbMap[9] = 3;
    GtCaloInputToPsbMap[10] = 3;



}

// destructor
L1GlobalTriggerReadoutSetup::~L1GlobalTriggerReadoutSetup()
{

    // nothing yet

}

boost::uint16_t L1GlobalTriggerReadoutSetup::GtBoard::boardId() const
{
    boost::uint16_t boardIdValue = 0;
    int boardHexName = 0;
    int boardSlot = 0;

    typedef std::map<L1GlobalTriggerReadoutSetup::GtBoard, int>::const_iterator CItBoardInt;
    typedef std::map<GtBoardType, int>::const_iterator CItBoardTypeInt;

    L1GlobalTriggerReadoutSetup tmpGtSetup; // TODO FIXME temporary event setup

    std::map<GtBoardType, int> hexNameMap = tmpGtSetup.GtBoardHexNameMap;
    std::map<L1GlobalTriggerReadoutSetup::GtBoard, int> slotMap = tmpGtSetup.GtBoardSlotMap;

    CItBoardTypeInt itHex = hexNameMap.find(boardType);
    if (itHex != hexNameMap.end()) {
        boardHexName = itHex->second;
    }

    CItBoardInt itSlot = slotMap.find(*this);
    if (itSlot != slotMap.end()) {
        boardSlot = itSlot->second;
    }

    if (boardType == GTFE) {
        boardIdValue = boardIdValue | boardSlot;
    } else {
        boardIdValue = boardIdValue | (boardHexName << 8) | boardSlot;
    }


    return boardIdValue;

}

bool L1GlobalTriggerReadoutSetup::GtBoard::operator< (
    const L1GlobalTriggerReadoutSetup::GtBoard& gtb) const
{
    if (boardType < gtb.boardType) {
        return true;
    } else {
        if (boardType == gtb.boardType) {

            if (boardIndex < gtb.boardIndex) {
                return true;
            }
        }
    }

    return false;
}
