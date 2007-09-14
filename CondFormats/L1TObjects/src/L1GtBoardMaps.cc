/**
 * \class L1GtBoardMaps
 * 
 * 
 * Description: various mappings of the L1 GT boards.  
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
#include "CondFormats/L1TObjects/interface/L1GtBoardMaps.h"

// system include files
#include <vector>
#include <ostream>

// user include files
#include "CondFormats/L1TObjects/interface/L1GtFwd.h"
#include "CondFormats/L1TObjects/interface/L1GtBoard.h"

// forward declarations

// constructor
L1GtBoardMaps::L1GtBoardMaps()
{
    // empty
}

// destructor
L1GtBoardMaps::~L1GtBoardMaps()
{
    // empty
}



// set / print L1 GT DAQ record map
void L1GtBoardMaps::setGtDaqRecordMap(const std::map<int, L1GtBoard>& gtDaqRecordMapValue)
{

    m_gtDaqRecordMap = gtDaqRecordMapValue;

}

void L1GtBoardMaps::printGtDaqRecordMap(std::ostream& s) const
{
    s << "\nL1 GT Trigger DAQ record map" << std::endl;

    s << "  Size: " << m_gtDaqRecordMap.size() << " board blocks." << std::endl;
    s << "  Header and trailer are automatically added to the hardware record.\n"
    << std::endl;

    for (CItIntBoard cIt = m_gtDaqRecordMap.begin(); cIt != m_gtDaqRecordMap.end(); ++cIt) {

        s << "  Position " << cIt->first
        << ": \t board " << cIt->second.boardName() << " " << cIt->second.boardIndex()
        << std::endl;
    }

    s << std::endl;

}


// set / print L1 GT EVM record map
void L1GtBoardMaps::setGtEvmRecordMap(const std::map<int, L1GtBoard>& gtEvmRecordMapValue)
{

    m_gtEvmRecordMap = gtEvmRecordMapValue;

}

void L1GtBoardMaps::printGtEvmRecordMap(std::ostream& s) const
{
    s << "\nL1 GT Trigger EVM record map" << std::endl;

    s << "  Size: " << m_gtEvmRecordMap.size() << " board blocks." << std::endl;
    s << "  Header and trailer are automatically added to the hardware record.\n"
    << std::endl;

    for (CItIntBoard cIt = m_gtEvmRecordMap.begin(); cIt != m_gtEvmRecordMap.end(); ++cIt) {

        s << "  Position " << cIt->first
        << ": \t board " << cIt->second.boardName() << " " << cIt->second.boardIndex()
        << std::endl;
    }

    s << std::endl;

}

// set / print L1 GT active boards map for DAQ record
void L1GtBoardMaps::setGtDaqActiveBoardsMap(
    const std::map<L1GtBoard, int>& gtDaqActiveBoardsMapValue)
{

    m_gtDaqActiveBoardsMap = gtDaqActiveBoardsMapValue;

}

void L1GtBoardMaps::printGtDaqActiveBoardsMap(std::ostream& s) const
{
    s << "\nL1 GT Trigger DAQ \"active boards\" record map" << std::endl;

    s << "  Size: " << m_gtDaqActiveBoardsMap.size() << " boards." << std::endl;
    s << std::endl;

    for (CItBoardInt cIt = m_gtDaqActiveBoardsMap.begin();
            cIt != m_gtDaqActiveBoardsMap.end(); ++cIt) {

        s << "  Bit " << cIt->second
        << ": \t board " << cIt->first.boardName() << " " << cIt->first.boardIndex()
        << std::endl;
    }

    s << std::endl;

}

// set / print L1 GT active boards map for EVM record
void L1GtBoardMaps::setGtEvmActiveBoardsMap(
    const std::map<L1GtBoard, int>& gtEvmActiveBoardsMapValue)
{

    m_gtEvmActiveBoardsMap = gtEvmActiveBoardsMapValue;

}

void L1GtBoardMaps::printGtEvmActiveBoardsMap(std::ostream& s) const
{
    s << "\nL1 GT Trigger EVM \"active boards\" record map" << std::endl;

    s << "  Size: " << m_gtEvmActiveBoardsMap.size() << " boards." << std::endl;
    s << std::endl;

    for (CItBoardInt cIt = m_gtEvmActiveBoardsMap.begin();
            cIt != m_gtEvmActiveBoardsMap.end(); ++cIt) {

        s << "  Bit " << cIt->second
        << ": \t board " << cIt->first.boardName() << " " << cIt->first.boardIndex()
        << std::endl;
    }

    s << std::endl;

}


// set / print L1 GT board - slot map
void L1GtBoardMaps::setGtBoardSlotMap(
    const std::map<L1GtBoard, int>& gtBoardSlotMapValue)
{

    m_gtBoardSlotMap = gtBoardSlotMapValue;

}

void L1GtBoardMaps::printGtBoardSlotMap(std::ostream& s) const
{
    s << "\nL1 GT Trigger board - slot map" << std::endl;

    s << "  Size: " << m_gtBoardSlotMap.size() << " boards." << std::endl;
    s << std::endl;

    for (CItBoardInt cIt = m_gtBoardSlotMap.begin();
            cIt != m_gtBoardSlotMap.end(); ++cIt) {

        s << "  Slot " << cIt->second
        << ": \t board " << cIt->first.boardName() << " " << cIt->first.boardIndex()
        << std::endl;
    }

    s << std::endl;

}

// set / print L1 GT board name in hw record map
void L1GtBoardMaps::setGtBoardHexNameMap(
    const std::map<L1GtBoard, int>& gtBoardHexNameMapValue)
{

    m_gtBoardHexNameMap = gtBoardHexNameMapValue;

}

void L1GtBoardMaps::printGtBoardHexNameMap(std::ostream& s) const
{
    s << "\nL1 GT Trigger board names in hw record map" << std::endl;

    s << "  Size: " << m_gtBoardHexNameMap.size() << " boards." << std::endl;
    s << std::endl;

    for (CItBoardInt cIt = m_gtBoardHexNameMap.begin();
            cIt != m_gtBoardHexNameMap.end(); ++cIt) {

        s << "  Hex name " << std::hex << cIt->second << std::dec
        << ": \t board " << cIt->first.boardName() << " " << cIt->first.boardIndex()
        << std::endl;
    }

    s << std::endl;

}

// set / print L1 GT calo input map
void L1GtBoardMaps::setGtCaloObjectInputMap(
    const std::map<int, L1GtCaloQuad>& gtCaloObjectInputMapValue)
{

    m_gtCaloObjectInputMap = gtCaloObjectInputMapValue;

}

void L1GtBoardMaps::printGtCaloObjectInputMap(std::ostream& s) const
{
    s << "\nL1 GT calorimeter input map" << std::endl;

    s << "  Size: " << m_gtCaloObjectInputMap.size() << " cables in this map." << std::endl;
    s << std::endl;

    for (CItIntCaloQ cIt = m_gtCaloObjectInputMap.begin();
            cIt != m_gtCaloObjectInputMap.end(); ++cIt) {

        std::string objType;

        if ( cIt->second == IsoEGQ ) {
            objType = "IsoEGQ";
        } else if ( cIt->second == NoIsoEGQ ) {
            objType = "NoIsoEGQ";
        } else if ( cIt->second == CenJetQ ) {
            objType = "CenJetQ";
        } else if ( cIt->second == ForJetQ ) {
            objType = "ForJetQ";
        } else if ( cIt->second == TauJetQ ) {
            objType = "TauJetQ";
        } else if ( cIt->second == ESumsQ ) {
            objType = "ESumsQ";
        } else if ( cIt->second == JetCountsQ ) {
            objType = "JetCountsQ";
        } else {
            // do nothing, return empty string
        }

        s << "  Cable CA_" << cIt->first
        << " input: \t " << objType
        << std::endl;
    }

    s << std::endl;

}

// set / print L1 GT calo input to PSB map

void L1GtBoardMaps::setGtCaloInputToPsbMap(
    const std::map<int, int>& gtCaloInputToPsbMapValue)
{

    m_gtCaloInputToPsbMap = gtCaloInputToPsbMapValue;

}

void L1GtBoardMaps::printGtCaloInputToPsbMap(std::ostream& s) const
{
    s << "\nL1 GT \"calorimeter input to PSB\" map" << std::endl;

    s << "  Size: " << m_gtCaloInputToPsbMap.size() << " cables." << std::endl;
    s << std::endl;

    for (CItIntInt cIt = m_gtCaloInputToPsbMap.begin();
            cIt != m_gtCaloInputToPsbMap.end(); ++cIt) {

        s << "  Cable CA_" << cIt->first
        << "\t belongs to PSB_" << cIt->second
        << std::endl;
    }

    s << std::endl;

}

// get the board ID - it needs the maps from event setup
const boost::uint16_t L1GtBoardMaps::boardId(const L1GtBoard& gtb) const
{

    boost::uint16_t boardIdValue = 0;

    int boardHexName = 0;
    int boardSlot = 0;

    CItBoardInt itHex = m_gtBoardHexNameMap.find(gtb);
    if (itHex != m_gtBoardHexNameMap.end()) {
        boardHexName = itHex->second;
    }

    CItBoardInt itSlot = m_gtBoardSlotMap.find(gtb);
    if (itSlot != m_gtBoardSlotMap.end()) {
        boardSlot = itSlot->second;
    }

    if (gtb.boardType() == GTFE) {
        boardIdValue = boardIdValue | boardSlot;
    } else {
        boardIdValue = boardIdValue | (boardHexName << 8) | boardSlot;
    }


    return boardIdValue;

}
