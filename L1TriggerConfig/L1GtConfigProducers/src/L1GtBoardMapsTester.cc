/**
 * \class L1GtBoardMapsTester
 * 
 * 
 * Description: test analyzer for various mappings of the L1 GT boards.  
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
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtBoardMapsTester.h"

// system include files
#include <map>
#include <iomanip>

// user include files
//   base class
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TObjects/interface/L1GtBoardMaps.h"
#include "CondFormats/DataRecord/interface/L1GtBoardMapsRcd.h"

// forward declarations

// constructor(s)
L1GtBoardMapsTester::L1GtBoardMapsTester(const edm::ParameterSet& parSet)
{
    // empty
}

// destructor
L1GtBoardMapsTester::~L1GtBoardMapsTester()
{
    // empty
}

// loop over events
void L1GtBoardMapsTester::analyze(
    const edm::Event& iEvent, const edm::EventSetup& evSetup)
{


    edm::ESHandle< L1GtBoardMaps > l1GtBM ;
    evSetup.get< L1GtBoardMapsRcd >().get( l1GtBM ) ;

    l1GtBM->printGtDaqRecordMap(std::cout);
    std::cout << std::endl;

    l1GtBM->printGtEvmRecordMap(std::cout);
    std::cout << std::endl;

    l1GtBM->printGtDaqActiveBoardsMap(std::cout);
    std::cout << std::endl;

    l1GtBM->printGtEvmActiveBoardsMap(std::cout);
    std::cout << std::endl;

    l1GtBM->printGtBoardSlotMap(std::cout);
    std::cout << std::endl;

    l1GtBM->printGtBoardHexNameMap(std::cout);
    std::cout << std::endl;

    l1GtBM->printGtCaloObjectInputMap(std::cout);
    std::cout << std::endl;

    l1GtBM->printGtCaloInputToPsbMap(std::cout);
    std::cout << std::endl;

    // print board ID

    std::map<L1GtBoard, int> boardList = l1GtBM->gtBoardHexNameMap();
    std::cout << "\nL1 GT Trigger: board ID " << std::endl;

    std::cout << "  Size: " << boardList.size() << " boards." << std::endl;
    std::cout << std::endl;

    typedef std::map<L1GtBoard, int>::const_iterator CItBoardInt;
    for (CItBoardInt cIt = boardList.begin(); cIt != boardList.end(); ++cIt) {

        std::cout << "  Board ID (hex): "
        << std::hex << l1GtBM->boardId(cIt->first) << std::dec
        << ": \t board " << cIt->first.boardName() << " " << cIt->first.boardIndex()
        << std::endl;
    }

    std::cout << std::endl;


}
