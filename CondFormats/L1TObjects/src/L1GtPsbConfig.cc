/**
 * \class L1GtPsbConfig
 *
 *
 * Description: class for L1 GT PSB board configuration.
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
#include "CondFormats/L1TObjects/interface/L1GtPsbConfig.h"

// system include files
#include <iostream>
#include <iomanip>

// user include files
//   base class

// forward declarations

// constructors
L1GtPsbConfig::L1GtPsbConfig()
{

    m_gtBoardSlot = -1;

    m_gtPsbCh0SendLvds = false;
    m_gtPsbCh1SendLvds = false;

    m_gtPsbEnableRecLvds.reserve(PsbNumberLvdsGroups);
    m_gtPsbEnableRecSerLink.reserve(PsbSerLinkNumberChannels);

}

// constructor using board slot
L1GtPsbConfig::L1GtPsbConfig(const int& psbSlot) :
    m_gtBoardSlot(psbSlot)
{

    m_gtPsbCh0SendLvds = false;
    m_gtPsbCh1SendLvds = false;

    m_gtPsbEnableRecLvds.reserve(PsbNumberLvdsGroups);
    m_gtPsbEnableRecSerLink.reserve(PsbSerLinkNumberChannels);

}

// destructor
L1GtPsbConfig::~L1GtPsbConfig()
{
    // empty
}

// copy constructor
L1GtPsbConfig::L1GtPsbConfig(const L1GtPsbConfig& gtb)
{

    m_gtBoardSlot = gtb.m_gtBoardSlot;

    m_gtPsbCh0SendLvds = gtb.m_gtPsbCh0SendLvds;
    m_gtPsbCh1SendLvds = gtb.m_gtPsbCh1SendLvds;

    m_gtPsbEnableRecLvds = gtb.m_gtPsbEnableRecLvds;
    m_gtPsbEnableRecSerLink = gtb.m_gtPsbEnableRecSerLink;

}

// assignment operator
L1GtPsbConfig& L1GtPsbConfig::operator=(const L1GtPsbConfig& gtb)
{

    if (this != &gtb) {

        m_gtBoardSlot = gtb.m_gtBoardSlot;

        m_gtPsbCh0SendLvds = gtb.m_gtPsbCh0SendLvds;
        m_gtPsbCh1SendLvds = gtb.m_gtPsbCh1SendLvds;

        m_gtPsbEnableRecLvds = gtb.m_gtPsbEnableRecLvds;
        m_gtPsbEnableRecSerLink = gtb.m_gtPsbEnableRecSerLink;
    }

    return *this;

}

// equal operator
bool L1GtPsbConfig::operator==(const L1GtPsbConfig& gtb) const
{

    if (m_gtBoardSlot != gtb.m_gtBoardSlot) {
        return false;
    }

    if (m_gtPsbCh0SendLvds != gtb.m_gtPsbCh0SendLvds) {
        return false;
    }

    if (m_gtPsbCh1SendLvds != gtb.m_gtPsbCh1SendLvds) {
        return false;
    }

    if (m_gtPsbEnableRecLvds != gtb.m_gtPsbEnableRecLvds) {
        return false;
    }

    if (m_gtPsbEnableRecSerLink != gtb.m_gtPsbEnableRecSerLink) {
        return false;
    }

    // all members identical
    return true;

}

// unequal operator
bool L1GtPsbConfig::operator!=(const L1GtPsbConfig& result) const
{

    return !(result == *this);

}

// less than operator
bool L1GtPsbConfig::operator<(const L1GtPsbConfig& gtb) const
{
    if (m_gtBoardSlot < gtb.m_gtBoardSlot) {
        return true;
    }
    else {
        return false;
    }

    return false;
}

// set board slot
void L1GtPsbConfig::setGtBoardSlot(const int& gtBoardSlotValue)
{
    m_gtBoardSlot = gtBoardSlotValue;
}

// set CH0_SEND_LVDS_NOT_DS92LV16
void L1GtPsbConfig::setGtPsbCh0SendLvds(const bool& gtPsbCh0SendLvdsValue)
{
    m_gtPsbCh0SendLvds = gtPsbCh0SendLvdsValue;
}

// set CH1_SEND_LVDS_NOT_DS92LV16
void L1GtPsbConfig::setGtPsbCh1SendLvds(const bool& gtPsbCh1SendLvdsValue)
{
    m_gtPsbCh1SendLvds = gtPsbCh1SendLvdsValue;
}

// set enable LVDS
void L1GtPsbConfig::setGtPsbEnableRecLvds(
        const std::vector<bool>& gtPsbEnableRecLvdsValue)
{

    m_gtPsbEnableRecLvds = gtPsbEnableRecLvdsValue;
}

// set enable channels for receiving signal via serial links
void L1GtPsbConfig::setGtPsbEnableRecSerLink(
        const std::vector<bool>& gtPsbEnableRecSerLinkValue)
{
    m_gtPsbEnableRecSerLink = gtPsbEnableRecSerLinkValue;
}

// print board
void L1GtPsbConfig::print(std::ostream& myCout) const
{

    myCout << "PSB Board slot " << m_gtBoardSlot << " ( 0x" << std::hex
            << m_gtBoardSlot << std::dec << " ):" << std::endl;

    myCout << "    CH0_SEND_LVDS_NOT_DS92LV16 = "
            << (m_gtPsbCh0SendLvds ? "True" : "False") << std::endl;
    myCout << "    CH1_SEND_LVDS_NOT_DS92LV16 = "
            << (m_gtPsbCh1SendLvds ? "True" : "False") << std::endl;
    myCout << std::endl;

    int iLvds = -1;
    for (std::vector<bool>::const_iterator cIt = m_gtPsbEnableRecLvds.begin(); cIt
            != m_gtPsbEnableRecLvds.end(); ++cIt) {

        iLvds++;
        myCout << "\n    Enable_Rec_LVDS [" << iLvds << "] = "
                << ((*cIt) ? "True" : "False");
    }
    myCout << std::endl;

    int iCh = -1;
    for (std::vector<bool>::const_iterator cIt =
            m_gtPsbEnableRecSerLink.begin(); cIt
            != m_gtPsbEnableRecSerLink.end(); ++cIt) {

        iCh++;
        myCout << "\n    SerLink_Ch" << iCh << "_Rec_Enable = "
                << ((*cIt) ? "True" : "False");
    }
    myCout << std::endl;
}

// output stream operator
std::ostream& operator<<(std::ostream& os, const L1GtPsbConfig& result)
{
    result.print(os);
    return os;

}

// number of LVDS groups per board
const int L1GtPsbConfig::PsbNumberLvdsGroups = 16;

// number of channels per board
const int L1GtPsbConfig::PsbSerLinkNumberChannels = 8;
