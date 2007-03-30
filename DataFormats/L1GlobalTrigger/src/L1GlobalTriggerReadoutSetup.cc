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
 * $Date:$
 * $Revision:$
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

    //populate the maps

    // L1 GT DAQ record map
    //    gives the position of each block in the GT DAQ readout record

    // header has position 0

    // GTFE position 1
    int iBoard = 1;

    GtBoard gtfeBoard = {GTFE, 0};
    GtDaqRecordMap[iBoard] = gtfeBoard;
    iBoard++;

    GtBoard fdlBoard = {FDL, 0};
    GtDaqRecordMap[iBoard] = fdlBoard;
    iBoard++;

    GtBoard psbBoard;
    psbBoard.boardType = PSB;
    for (int iPsb = 0; iPsb < NumberPsbBoards; ++iPsb) {
        psbBoard.boardIndex = iPsb;
        GtDaqRecordMap[iBoard] = psbBoard;
        iBoard++;
    }

    GtBoard gmtBoard = {GMT, 0};
    GtDaqRecordMap[iBoard] = gmtBoard;



    // L1 GT active boards map
    //    gives the bit of each GT board in the GTFE ACTIVE_BOARDS
    //    for the GT DAQ readout record

    int iBit = 0;

    GtDaqActiveBoardsMap[fdlBoard] = iBit;
    iBit++;

    for (int iPsb = 0; iPsb < NumberPsbBoards; ++iPsb) {
        psbBoard.boardIndex = iBit;
        GtDaqActiveBoardsMap[psbBoard] = iBit;
        iBit++;
    }


    GtDaqActiveBoardsMap[gmtBoard] = iBit;
    iBit++;

    gmtBoard.boardIndex = 1; // spare, not used

    GtDaqActiveBoardsMap[gmtBoard] = iBit;
    iBit++;

    GtBoard tcsBoard = {TCS, 0};

    GtDaqActiveBoardsMap[tcsBoard] = iBit;
    iBit++;

    GtBoard timBoard = {TIM, 0};

    GtDaqActiveBoardsMap[timBoard] = iBit;
    iBit++;


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
