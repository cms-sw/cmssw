#ifndef CondFormats_L1TObjects_L1GtBoardMaps_h
#define CondFormats_L1TObjects_L1GtBoardMaps_h

/**
 * \class L1GtBoardMaps
 *
 *
 * Description: map of the L1 GT boards.
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
#include "CondFormats/L1TObjects/interface/L1GtBoard.h"

// forward declarations

// class declaration
class L1GtBoardMaps
{

public:

    // constructor
    L1GtBoardMaps();

    // destructor
    virtual ~L1GtBoardMaps();

public:

    /// get / set / print the L1 GT board map
    const std::vector<L1GtBoard>& gtBoardMaps() const
    {
        return m_gtBoardMaps;
    }

    void setGtBoardMaps(const std::vector<L1GtBoard>&);
    void print(std::ostream&) const;

    /// output stream operator
    friend std::ostream& operator<<(std::ostream&, const L1GtBoardMaps&);


public:

    /// print L1 GT DAQ record map
    void printGtDaqRecordMap(std::ostream& myCout) const;

    /// print L1 GT EVM record map
    void printGtEvmRecordMap(std::ostream& myCout) const;

    /// print L1 GT active boards map for DAQ record
    void printGtDaqActiveBoardsMap(std::ostream& myCout) const;

    /// print L1 GT active boards map for EVM record
    void printGtEvmActiveBoardsMap(std::ostream& myCout) const;

    /// print L1 GT board - slot map
    void printGtBoardSlotMap(std::ostream& myCout) const;

    /// print L1 GT board name in hw record map
    void printGtBoardHexNameMap(std::ostream& myCout) const;

    /// print L1 quadruplet (4x16 bits)(cable) to PSB input map
    void printGtQuadToPsbMap(std::ostream& myCout) const;

private:

    /// L1 GT boards and their mapping
    std::vector<L1GtBoard> m_gtBoardMaps;


    COND_SERIALIZABLE;
};

#endif /*CondFormats_L1TObjects_L1GtBoardMaps_h*/
