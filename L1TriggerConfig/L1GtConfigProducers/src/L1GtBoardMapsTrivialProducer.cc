/**
 * \class L1GtBoardMapsTrivialProducer
 * 
 * 
 * Description: ESProducer for various mappings of the L1 GT boards.  
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
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtBoardMapsTrivialProducer.h"

// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

#include <string>
#include <vector>
#include <map>


// user include files
//   base class
#include "FWCore/Utilities/interface/EDMException.h"

#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TObjects/interface/L1GtFwd.h"
#include "CondFormats/L1TObjects/interface/L1GtBoard.h"
#include "CondFormats/DataRecord/interface/L1GtBoardMapsRcd.h"

// forward declarations

// constructor(s)
L1GtBoardMapsTrivialProducer::L1GtBoardMapsTrivialProducer(const edm::ParameterSet& parSet)
{
    // tell the framework what data is being produced
    setWhatProduced(this, &L1GtBoardMapsTrivialProducer::produceBoardMaps);

    // now do what ever other initialization is needed

    // get the list of the board names and indices
    std::vector<std::string> boardList =
        parSet.getParameter<std::vector<std::string> >("BoardList");

    std::vector<int> boardIndexVec =
        parSet.getParameter<std::vector<int> >("BoardIndex");

    // check if the board list and the board indices are consistent
    // i.e. have the same number of entries

    if (boardList.size() != boardIndexVec.size()) {
        throw cms::Exception("Configuration")
        << "\nError: inconsistent length of board list and board indices.\n"
        << "\n       Can not define the mapping of the L1 GT boards.     \n"
        << std::endl;
    }

    // L1 GT DAQ record map
    std::vector<int> boardIndexDaqRecord =
        parSet.getParameter<std::vector<int> >("BoardIndexDaqRecord");

    if (boardList.size() != boardIndexDaqRecord.size()) {
        throw cms::Exception("Configuration")
        << "\nError: inconsistent length of board list and board indices in GT DAQ record.\n"
        << "\n       Can not define the mapping of the L1 GT boards.     \n"
        << std::endl;
    }

    // L1 GT EVM record map
    std::vector<int> boardIndexEvmRecord =
        parSet.getParameter<std::vector<int> >("BoardIndexEvmRecord");

    if (boardList.size() != boardIndexEvmRecord.size()) {
        throw cms::Exception("Configuration")
        << "\nError: inconsistent length of board list and board indices in GT EVM record.\n"
        << "\n       Can not define the mapping of the L1 GT boards.     \n"
        << std::endl;
    }

    // L1 GT "active boards" map for DAQ record
    std::vector<int> activeBoardsDaqRecord =
        parSet.getParameter<std::vector<int> >("ActiveBoardsDaqRecord");

    if (boardList.size() != activeBoardsDaqRecord.size()) {
        throw cms::Exception("Configuration")
        << "\nError: inconsistent length of board list and active boards in GT DAQ record.\n"
        << "\n       Can not define the mapping of the L1 GT boards.     \n"
        << std::endl;
    }

    // L1 GT "active boards" map for EVM record
    std::vector<int> activeBoardsEvmRecord =
        parSet.getParameter<std::vector<int> >("ActiveBoardsEvmRecord");

    if (boardList.size() != activeBoardsEvmRecord.size()) {
        throw cms::Exception("Configuration")
        << "\nError: inconsistent length of board list and active boards in GT EVM record.\n"
        << "\n       Can not define the mapping of the L1 GT boards.     \n"
        << std::endl;
    }

    // L1 GT board - slot map
    std::vector<int> boardSlotMap =
        parSet.getParameter<std::vector<int> >("BoardSlotMap");

    if (boardList.size() != boardSlotMap.size()) {
        throw cms::Exception("Configuration")
        << "\nError: inconsistent length of board list and board - slot map.\n"
        << "\n       Can not define the mapping of the L1 GT boards.     \n"
        << std::endl;
    }

    // L1 GT board name in hw record map
    std::vector<int> boardHexNameMap =
        parSet.getParameter<std::vector<int> >("BoardHexNameMap");

    if (boardList.size() != boardHexNameMap.size()) {
        throw cms::Exception("Configuration")
        << "\nError: inconsistent length of board list and board name in hw record map.\n"
        << "\n       Can not define the mapping of the L1 GT boards.     \n"
        << std::endl;
    }


    // fill the maps

    int posVec = 0;

    for (std::vector<std::string>::const_iterator it = boardList.begin();
            it != boardList.end(); ++it) {

        if ( (*it) == "GTFE" ) {

            int iBoard = boardIndexVec.at(posVec);
            L1GtBoard gtfeBoard = L1GtBoard(GTFE, iBoard);

            // entry in L1 GT DAQ record map
            iBoard = boardIndexDaqRecord.at(posVec);
            if (iBoard >= 0) {
                m_gtDaqRecordMap[iBoard] = gtfeBoard;
            }

            // entry in L1 GT EVM record map
            iBoard = boardIndexEvmRecord.at(posVec);
            if (iBoard >= 0) {
                m_gtEvmRecordMap[iBoard] = gtfeBoard;
            }

            // entry in "active boards" map for DAQ record
            iBoard = activeBoardsDaqRecord.at(posVec);
            if (iBoard >= 0) {
                m_gtDaqActiveBoardsMap[gtfeBoard] = iBoard;
            }

            // entry in "active boards" map for EVM record
            iBoard = activeBoardsEvmRecord.at(posVec);
            if (iBoard >= 0) {
                m_gtEvmActiveBoardsMap[gtfeBoard] = iBoard;
            }

            // entry in board - slot map
            iBoard = boardSlotMap.at(posVec);
            if (iBoard >= 0) {
                m_gtBoardSlotMap[gtfeBoard] = iBoard;
            }

            // entry in board name in hw record map
            iBoard = boardHexNameMap.at(posVec);
            if (iBoard >= 0) {
                m_gtBoardHexNameMap[gtfeBoard] = iBoard;
            }

        } else if ( (*it) == "FDL" ) {

            int iBoard = boardIndexVec.at(posVec);
            L1GtBoard fdlBoard = L1GtBoard(FDL, iBoard);

            // entry in L1 GT DAQ record map
            iBoard = boardIndexDaqRecord.at(posVec);
            if (iBoard >= 0) {
                m_gtDaqRecordMap[iBoard] = fdlBoard;
            }

            // entry in L1 GT EVM record map
            iBoard = boardIndexEvmRecord.at(posVec);
            if (iBoard >= 0) {
                m_gtEvmRecordMap[iBoard] = fdlBoard;
            }

            // entry in "active boards" map for DAQ record
            iBoard = activeBoardsDaqRecord.at(posVec);
            if (iBoard >= 0) {
                m_gtDaqActiveBoardsMap[fdlBoard] = iBoard;
            }

            // entry in "active boards" map for EVM record
            iBoard = activeBoardsEvmRecord.at(posVec);
            if (iBoard >= 0) {
                m_gtEvmActiveBoardsMap[fdlBoard] = iBoard;
            }

            // entry in board - slot map
            iBoard = boardSlotMap.at(posVec);
            if (iBoard >= 0) {
                m_gtBoardSlotMap[fdlBoard] = iBoard;
            }

            // entry in board name in hw record map
            iBoard = boardHexNameMap.at(posVec);
            if (iBoard >= 0) {
                m_gtBoardHexNameMap[fdlBoard] = iBoard;
            }

        } else if ( (*it) == "PSB" ) {

            int iBoard = boardIndexVec.at(posVec);
            L1GtBoard psbBoard = L1GtBoard(PSB, iBoard);

            // entry in L1 GT DAQ record map
            iBoard = boardIndexDaqRecord.at(posVec);
            if (iBoard >= 0) {
                m_gtDaqRecordMap[iBoard] = psbBoard;
            }

            // entry in L1 GT EVM record map
            iBoard = boardIndexEvmRecord.at(posVec);
            if (iBoard >= 0) {
                m_gtEvmRecordMap[iBoard] = psbBoard;
            }

            // entry in "active boards" map for DAQ record
            iBoard = activeBoardsDaqRecord.at(posVec);
            if (iBoard >= 0) {
                m_gtDaqActiveBoardsMap[psbBoard] = iBoard;
            }

            // entry in "active boards" map for EVM record
            iBoard = activeBoardsEvmRecord.at(posVec);
            if (iBoard >= 0) {
                m_gtEvmActiveBoardsMap[psbBoard] = iBoard;
            }

            // entry in board - slot map
            iBoard = boardSlotMap.at(posVec);
            if (iBoard >= 0) {
                m_gtBoardSlotMap[psbBoard] = iBoard;
            }

            // entry in board name in hw record map
            iBoard = boardHexNameMap.at(posVec);
            if (iBoard >= 0) {
                m_gtBoardHexNameMap[psbBoard] = iBoard;
            }

        } else if ( (*it) == "GMT" ) {

            int iBoard = boardIndexVec.at(posVec);
            L1GtBoard gmtBoard = L1GtBoard(GMT, iBoard);

            // entry in L1 GT DAQ record map
            iBoard = boardIndexDaqRecord.at(posVec);
            if (iBoard >= 0) {
                m_gtDaqRecordMap[iBoard] = gmtBoard;
            }

            // entry in L1 GT EVM record map
            iBoard = boardIndexEvmRecord.at(posVec);
            if (iBoard >= 0) {
                m_gtEvmRecordMap[iBoard] = gmtBoard;
            }

            // entry in "active boards" map for DAQ record
            iBoard = activeBoardsDaqRecord.at(posVec);
            if (iBoard >= 0) {
                m_gtDaqActiveBoardsMap[gmtBoard] = iBoard;
            }

            // entry in "active boards" map for EVM record
            iBoard = activeBoardsEvmRecord.at(posVec);
            if (iBoard >= 0) {
                m_gtEvmActiveBoardsMap[gmtBoard] = iBoard;
            }

            // entry in board - slot map
            iBoard = boardSlotMap.at(posVec);
            if (iBoard >= 0) {
                m_gtBoardSlotMap[gmtBoard] = iBoard;
            }

            // entry in board name in hw record map
            iBoard = boardHexNameMap.at(posVec);
            if (iBoard >= 0) {
                m_gtBoardHexNameMap[gmtBoard] = iBoard;
            }

        } else if ( (*it) == "TCS" ) {

            int iBoard = boardIndexVec.at(posVec);
            L1GtBoard tcsBoard = L1GtBoard(TCS, iBoard);

            // entry in L1 GT DAQ record map
            iBoard = boardIndexDaqRecord.at(posVec);
            if (iBoard >= 0) {
                m_gtDaqRecordMap[iBoard] = tcsBoard;
            }

            // entry in L1 GT EVM record map
            iBoard = boardIndexEvmRecord.at(posVec);
            if (iBoard >= 0) {
                m_gtEvmRecordMap[iBoard] = tcsBoard;
            }

            // entry in "active boards" map for DAQ record
            iBoard = activeBoardsDaqRecord.at(posVec);
            if (iBoard >= 0) {
                m_gtDaqActiveBoardsMap[tcsBoard] = iBoard;
            }

            // entry in "active boards" map for EVM record
            iBoard = activeBoardsEvmRecord.at(posVec);
            if (iBoard >= 0) {
                m_gtEvmActiveBoardsMap[tcsBoard] = iBoard;
            }

            // entry in board - slot map
            iBoard = boardSlotMap.at(posVec);
            if (iBoard >= 0) {
                m_gtBoardSlotMap[tcsBoard] = iBoard;
            }

            // entry in board name in hw record map
            iBoard = boardHexNameMap.at(posVec);
            if (iBoard >= 0) {
                m_gtBoardHexNameMap[tcsBoard] = iBoard;
            }

        } else if ( (*it) == "TIM" ) {

            int iBoard = boardIndexVec.at(posVec);
            L1GtBoard timBoard = L1GtBoard(TIM, iBoard);

            // entry in L1 GT DAQ record map
            iBoard = boardIndexDaqRecord.at(posVec);
            if (iBoard >= 0) {
                m_gtDaqRecordMap[iBoard] = timBoard;
            }

            // entry in L1 GT EVM record map
            iBoard = boardIndexEvmRecord.at(posVec);
            if (iBoard >= 0) {
                m_gtEvmRecordMap[iBoard] = timBoard;
            }

            // entry in "active boards" map for DAQ record
            iBoard = activeBoardsDaqRecord.at(posVec);
            if (iBoard >= 0) {
                m_gtDaqActiveBoardsMap[timBoard] = iBoard;
            }

            // entry in "active boards" map for EVM record
            iBoard = activeBoardsEvmRecord.at(posVec);
            if (iBoard >= 0) {
                m_gtEvmActiveBoardsMap[timBoard] = iBoard;
            }

            // entry in board - slot map
            iBoard = boardSlotMap.at(posVec);
            if (iBoard >= 0) {
                m_gtBoardSlotMap[timBoard] = iBoard;
            }

            // entry in board name in hw record map
            iBoard = boardHexNameMap.at(posVec);
            if (iBoard >= 0) {
                m_gtBoardHexNameMap[timBoard] = iBoard;
            }

        } else {

            throw cms::Exception("Configuration")
            << "\nError: no such board: " << (*it).c_str() << "\n"
            << "\n       Can not define the mapping of the L1 GT boards.     \n"
            << std::endl;

        }

        posVec++;

    }


    // GCT to GT maps


    // L1 GT CA list (GCT input to PSB)
    std::vector<int> cableList =
        parSet.getParameter<std::vector<int> >("CableList");

    // L1 GT calo input map (IsoEGQ to CA_X, ...)
    //    gives the mapping of calorimeter objects to GT calorimeter input
    // GT calorimeter input will be mapped to PSBs later
    std::vector<std::string> caloObjectInputMap =
        parSet.getParameter<std::vector<std::string> >("CaloObjectInputMap");

    // check if the CA list and the calorimeter input are consistent
    // i.e. have the same number of entries

    if (cableList.size() != caloObjectInputMap.size()) {
        throw cms::Exception("Configuration")
        << "\nError: inconsistent length of cable list and GCT quadruplet list.\n"
        << "\n       Can not define the mapping of GCT quadruplets to GT cables.\n"
        << std::endl;
    }

    // L1 GT calo input to PSB map
    //    gives the mapping of GT calorimeter input to GT PSBs via PSB index
    //    4 infinicables per PSB (last PSB can use only 2!)
    std::vector<int> caloInputToPsbMap =
        parSet.getParameter<std::vector<int> >("CaloInputToPsbMap");


    if (cableList.size() != caloInputToPsbMap.size()) {
        throw cms::Exception("Configuration")
        << "\nError: inconsistent length of cable list and input to PSB list.\n"
        << "\n       Can not define the mapping of GCT quadruplets to GT PSBs.\n"
        << std::endl;
    }

    // fill the calorimeter related maps

    posVec = 0;

    for (std::vector<int>::const_iterator itCable = cableList.begin();
            itCable != cableList.end(); ++itCable) {

        std::string objType = caloObjectInputMap.at(posVec);

        if ( objType == "IsoEGQ" ) {
            m_gtCaloObjectInputMap[(*itCable)] = IsoEGQ;
        } else if ( objType == "NoIsoEGQ" ) {
            m_gtCaloObjectInputMap[(*itCable)] = NoIsoEGQ;
        } else if ( objType == "CenJetQ" ) {
            m_gtCaloObjectInputMap[(*itCable)] = CenJetQ;
        } else if ( objType == "ForJetQ" ) {
            m_gtCaloObjectInputMap[(*itCable)] = ForJetQ;
        } else if ( objType == "TauJetQ" ) {
            m_gtCaloObjectInputMap[(*itCable)] = TauJetQ;
        } else if ( objType == "ESumsQ" ) {
            m_gtCaloObjectInputMap[(*itCable)] = ESumsQ;
        } else if ( objType == "JetCountsQ" ) {
            m_gtCaloObjectInputMap[(*itCable)] = JetCountsQ;
        } else if ( objType == "Free" ) {
            // do nothing, no connection to that cable - cable free
        } else {
            throw cms::Exception("Configuration")
            << "\nError: no such GCT quadruplet: " << objType << "\n"
            << "\n       Can not define the mapping of the L1 GT calorimeter mapping.\n"
            << std::endl;
        }

        int iPSB = caloInputToPsbMap.at(posVec);
        m_gtCaloInputToPsbMap[(*itCable)] = iPSB;

        posVec++;

    }

}

// destructor
L1GtBoardMapsTrivialProducer::~L1GtBoardMapsTrivialProducer()
{

    // empty

}


// member functions

// method called to produce the data
boost::shared_ptr<L1GtBoardMaps> L1GtBoardMapsTrivialProducer::produceBoardMaps(
    const L1GtBoardMapsRcd& iRecord)
{

    using namespace edm::es;

    boost::shared_ptr<L1GtBoardMaps> pL1GtBoardMaps =
        boost::shared_ptr<L1GtBoardMaps>( new L1GtBoardMaps() );

    pL1GtBoardMaps->setGtDaqRecordMap(m_gtDaqRecordMap);
    pL1GtBoardMaps->setGtEvmRecordMap(m_gtEvmRecordMap);
    pL1GtBoardMaps->setGtDaqActiveBoardsMap(m_gtDaqActiveBoardsMap);
    pL1GtBoardMaps->setGtEvmActiveBoardsMap(m_gtEvmActiveBoardsMap);
    pL1GtBoardMaps->setGtBoardSlotMap(m_gtBoardSlotMap);
    pL1GtBoardMaps->setGtBoardHexNameMap(m_gtBoardHexNameMap);

    pL1GtBoardMaps->setGtCaloObjectInputMap(m_gtCaloObjectInputMap);
    pL1GtBoardMaps->setGtCaloInputToPsbMap(m_gtCaloInputToPsbMap);

    return pL1GtBoardMaps ;
}
