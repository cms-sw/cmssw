#ifndef CondFormats_L1TObjects_L1GtPsbConfig_h
#define CondFormats_L1TObjects_L1GtPsbConfig_h

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

// system include files
#include "CondFormats/Serialization/interface/Serializable.h"

#include <vector>
#include <iosfwd>

// user include files
#include "CondFormats/L1TObjects/interface/L1GtFwd.h"

// forward declarations

// class declaration
class L1GtPsbConfig
{

public:

    /// constructors
    L1GtPsbConfig();

    /// constructor using board slot
    L1GtPsbConfig(const int&);

    /// destructor
    virtual ~L1GtPsbConfig();

    /// copy constructor
    L1GtPsbConfig(const L1GtPsbConfig&);

    /// assignment operator
    L1GtPsbConfig& operator=(const L1GtPsbConfig&);

    /// equal operator
    bool operator==(const L1GtPsbConfig&) const;

    /// unequal operator
    bool operator!=(const L1GtPsbConfig&) const;

    /// less than operator
    bool operator<(const L1GtPsbConfig&) const;

public:

    /// number of LVDS groups per board
    static const int PsbNumberLvdsGroups;

    /// number of channels per board
    static const int PsbSerLinkNumberChannels;

public:

    /// get / set board slot
    inline const int gtBoardSlot() const
    {
        return m_gtBoardSlot;
    }

    void setGtBoardSlot(const int&);

    /// get / set CH0_SEND_LVDS_NOT_DS92LV16
    inline const bool gtPsbCh0SendLvds() const
    {
        return m_gtPsbCh0SendLvds;
    }

    void setGtPsbCh0SendLvds(const bool&);

    /// get / set CH1_SEND_LVDS_NOT_DS92LV16
    inline const bool gtPsbCh1SendLvds() const
    {
        return m_gtPsbCh1SendLvds;
    }

    void setGtPsbCh1SendLvds(const bool&);

    /// get / set enable LVDS
    inline const std::vector<bool>& gtPsbEnableRecLvds() const
    {
        return m_gtPsbEnableRecLvds;
    }

    void setGtPsbEnableRecLvds(const std::vector<bool>&);

    /// get / set enable channels for receiving signal via serial links
    inline const std::vector<bool>& gtPsbEnableRecSerLink() const
    {
        return m_gtPsbEnableRecSerLink;
    }

    void setGtPsbEnableRecSerLink(const std::vector<bool>&);

    /// print board
    void print(std::ostream& myCout) const;

    /// output stream operator
    friend std::ostream& operator<<(std::ostream&, const L1GtPsbConfig&);

private:

    /// the slot of board (part of Board_Id)
    int m_gtBoardSlot;

    /// CH0_SEND_LVDS_NOT_DS92LV16
    bool m_gtPsbCh0SendLvds;

    /// CH1_SEND_LVDS_NOT_DS92LV16
    bool m_gtPsbCh1SendLvds;

    /// enable LVDS (PsbNumberLvdsGroups = 16 groups of four bits)
    /// can be enabled/disabled per group
    std::vector<bool> m_gtPsbEnableRecLvds;

    /// enable channels for receiving signal via serial links
    std::vector<bool> m_gtPsbEnableRecSerLink;


    COND_SERIALIZABLE;
};

#endif /*CondFormats_L1TObjects_L1GtPsbConfig_h*/
