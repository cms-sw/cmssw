#ifndef L1GtConfigProducers_L1GtBoardMapsTrivialProducer_h
#define L1GtConfigProducers_L1GtBoardMapsTrivialProducer_h

/**
 * \class L1GtBoardMapsTrivialProducer
 * 
 * 
 * Description: ESProducer for mappings of the L1 GT boards.  
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
#include <memory>
#include "boost/shared_ptr.hpp"

#include <string>
#include <map>

// user include files
//   base class
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TObjects/interface/L1GtBoard.h"
#include "CondFormats/L1TObjects/interface/L1GtBoardMaps.h"
#include "CondFormats/DataRecord/interface/L1GtBoardMapsRcd.h"

// forward declarations

// class declaration
class L1GtBoardMapsTrivialProducer : public edm::ESProducer
{

public:

    /// constructor
    L1GtBoardMapsTrivialProducer(const edm::ParameterSet&);

    /// destructor
    ~L1GtBoardMapsTrivialProducer();


    /// public methods

    /// produce mappings of the L1 GT boards
    boost::shared_ptr<L1GtBoardMaps> produceBoardMaps(
        const L1GtBoardMapsRcd&);

private:

    /// L1 GT DAQ record map
    ///    gives the position of each block in the GT DAQ readout record
    std::map<int, L1GtBoard> m_gtDaqRecordMap;

    /// L1 GT EVM record map
    ///    gives the position of each block in the GT EVM readout record
    std::map<int, L1GtBoard> m_gtEvmRecordMap;

    /// L1 GT active boards map for DAQ record
    ///    gives the bit of each GT board in the GTFE ACTIVE_BOARDS
    ///    for the GT DAQ readout record
    std::map<L1GtBoard, int> m_gtDaqActiveBoardsMap;

    /// L1 GT active boards map for EVM record
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

#endif
