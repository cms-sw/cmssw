#ifndef CondFormats_L1TObjects_L1GtBoardMaps_h
#define CondFormats_L1TObjects_L1GtBoardMaps_h

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

// system include files
#include <string>
#include <vector>
#include <map>
#include <ostream>

#include <boost/cstdint.hpp>

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

    // constant iterators typedefs
    typedef std::map<L1GtBoard, int>::const_iterator CItBoardInt;
    typedef std::map<int, L1GtBoard>::const_iterator CItIntBoard;

    typedef std::map<int, int>::const_iterator CItIntInt;

    typedef std::map<int, L1GtCaloQuad>::const_iterator CItIntCaloQ;

public:

    // hardware-related stuff

    /// get / set / print L1 GT DAQ record map
    const std::map<int, L1GtBoard> gtDaqRecordMap() const
    {
        return m_gtDaqRecordMap;
    }

    void setGtDaqRecordMap(const std::map<int, L1GtBoard>&);
    void printGtDaqRecordMap(std::ostream&) const;

    /// get / set / print L1 GT EVM record map
    const std::map<int, L1GtBoard> gtEvmRecordMap() const
    {
        return m_gtEvmRecordMap;
    }

    void setGtEvmRecordMap(const std::map<int, L1GtBoard>&);
    void printGtEvmRecordMap(std::ostream&) const;

    /// get / set / print L1 GT active boards map for DAQ record
    const std::map<L1GtBoard, int> gtDaqActiveBoardsMap() const
    {
        return m_gtDaqActiveBoardsMap;
    }

    void setGtDaqActiveBoardsMap(const std::map<L1GtBoard, int>&);
    void printGtDaqActiveBoardsMap(std::ostream&) const;

    /// get / set / print L1 GT active boards map for EVM record
    const std::map<L1GtBoard, int> gtEvmActiveBoardsMap() const
    {
        return m_gtEvmActiveBoardsMap;
    }

    void setGtEvmActiveBoardsMap(const std::map<L1GtBoard, int>&);
    void printGtEvmActiveBoardsMap(std::ostream&) const;

    /// get / set / print L1 GT board - slot map
    const std::map<L1GtBoard, int> gtBoardSlotMap() const
    {
        return m_gtBoardSlotMap;
    }

    void setGtBoardSlotMap(const std::map<L1GtBoard, int>&);
    void printGtBoardSlotMap(std::ostream&) const;

    /// get / set / print L1 GT board name in hw record map
    const std::map<L1GtBoard, int> gtBoardHexNameMap() const
    {
        return m_gtBoardHexNameMap;
    }

    void setGtBoardHexNameMap(const std::map<L1GtBoard, int>&);
    void printGtBoardHexNameMap(std::ostream&) const;


    /// get / set / print L1 GT calo input map
    const std::map<int, L1GtCaloQuad> gtCaloObjectInputMap() const
    {
        return m_gtCaloObjectInputMap;
    }

    void setGtCaloObjectInputMap(const std::map<int, L1GtCaloQuad>&);
    void printGtCaloObjectInputMap(std::ostream&) const;


    /// get / set / print L1 GT calo input to PSB map
    const std::map<int, int> gtCaloInputToPsbMap() const
    {
        return m_gtCaloInputToPsbMap;
    }

    void setGtCaloInputToPsbMap(const std::map<int, int>&);
    void printGtCaloInputToPsbMap(std::ostream&) const;

    /// get the board ID - it needs the maps from event setup
    const boost::uint16_t boardId(const L1GtBoard&) const;



private:

    /// L1 GT DAQ record map
    ///    gives the position of each block in the GT DAQ readout record
    std::map<int, L1GtBoard> m_gtDaqRecordMap;

    /// L1 GT EVM record map
    ///    gives the position of each block in the GT EVM readout record
    std::map<int, L1GtBoard> m_gtEvmRecordMap;

    /// L1 GT active boards map
    ///    gives the bit of each GT board in the GTFE ACTIVE_BOARDS
    ///    for the GT DAQ readout record
    std::map<L1GtBoard, int> m_gtDaqActiveBoardsMap;

    /// L1 GT active boards map
    ///    gives the bit of each GT board in the GTFE ACTIVE_BOARDS
    ///    for the GT EVM readout record
    std::map<L1GtBoard, int> m_gtEvmActiveBoardsMap;

    /// L1 GT board - slot map
    ///    gives the slot of each GT board (part of Board_Id)
    std::map<L1GtBoard, int> m_gtBoardSlotMap;

    /// L1 GT board name in hw record map
    ///    gives the bits written for each GT board in the Board_Id
    std::map<L1GtBoard, int> m_gtBoardHexNameMap;


    /// L1 GT calo input map
    ///    gives the mapping of calorimeter objects to GT calorimeter input
    /// GT calorimeter input will be mapped to PSBs later
    std::map<int, L1GtCaloQuad> m_gtCaloObjectInputMap;


    /// L1 GT calo input to PSB map
    ///    gives the mapping of GT calorimeter input to GT PSBs
    std::map<int, int> m_gtCaloInputToPsbMap;




};

#endif /*CondFormats_L1TObjects_L1GtBoardMaps_h*/
