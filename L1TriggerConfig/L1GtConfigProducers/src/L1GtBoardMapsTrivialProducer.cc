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
 *
 */

// this class header
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtBoardMapsTrivialProducer.h"

// system include files
#include <memory>

#include <string>

// user include files
//   base class
#include "FWCore/Utilities/interface/EDMException.h"

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TObjects/interface/L1GtFwd.h"

// forward declarations

//

std::vector<L1GtObject> chInputObjects(
        const std::vector<std::string>& chInputStrings)
{
    std::vector<L1GtObject> chInputObjectsV;
    chInputObjectsV.reserve(chInputStrings.size());

    L1GtObject obj;

    for (std::vector<std::string>::const_iterator itObj =
            chInputStrings.begin(); itObj != chInputStrings.end(); ++itObj) {

        if ((*itObj) == "Mu") {
            obj = Mu;
        }
        else if ((*itObj) == "NoIsoEG") {
            obj = NoIsoEG;
        }
        else if ((*itObj) == "IsoEG") {
            obj = IsoEG;
        }
        else if ((*itObj) == "CenJet") {
            obj = CenJet;
        }
        else if ((*itObj) == "ForJet") {
            obj = ForJet;
        }
        else if ((*itObj) == "TauJet") {
            obj = TauJet;
        }
        else if ((*itObj) == "ETM") {
            obj = ETM;
        }
        else if ((*itObj) == "ETT") {
            obj = ETT;
        }
        else if ((*itObj) == "HTT") {
            obj = HTT;
        }
        else if ((*itObj) == "HTM") {
            obj = HTM;
        }
        else if ((*itObj) == "JetCounts") {
            obj = JetCounts;
        }
        else if ((*itObj) == "HfBitCounts") {
            obj = HfBitCounts;
        }
        else if ((*itObj) == "HfRingEtSums") {
            obj = HfRingEtSums;
        }
        else if ((*itObj) == "TechTrig") {
            obj = TechTrig;
        }
        else if ((*itObj) == "BPTX") {
            obj = BPTX;
        }
        else if ((*itObj) == "GtExternal") {
            obj = GtExternal;
        }
        else {
            throw cms::Exception("Configuration")
                    << "\nError: no such L1 GT object: " << (*itObj) << "\n"
                    << "\n       Can not define the mapping of the L1 GT boards.     \n"
                    << std::endl;

        }

        chInputObjectsV.push_back(obj);
    }

    return chInputObjectsV;
}

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
    std::vector<int> boardPositionDaqRecord =
        parSet.getParameter<std::vector<int> >("BoardPositionDaqRecord");

    if (boardList.size() != boardPositionDaqRecord.size()) {
        throw cms::Exception("Configuration")
        << "\nError: inconsistent length of board list and board indices in GT DAQ record.\n"
        << "\n       Can not define the mapping of the L1 GT boards.     \n"
        << std::endl;
    }

    // L1 GT EVM record map
    std::vector<int> boardPositionEvmRecord =
        parSet.getParameter<std::vector<int> >("BoardPositionEvmRecord");

    if (boardList.size() != boardPositionEvmRecord.size()) {
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



    // GCT PSB to GT - map cables to input quadruplets  and PSB indices


    // L1 GT cable list (GCT input to PSB)
    std::vector<std::string> cableList =
        parSet.getParameter<std::vector<std::string> >("CableList");

    // L1 GT calo input to PSB map
    //    gives the mapping of GT calorimeter input to GT PSBs via PSB index
    //    4 infinicables per PSB (last PSB can use only 2!)
    std::vector<int> cableToPsbMap =
        parSet.getParameter<std::vector<int> >("CableToPsbMap");


    if (cableList.size() != cableToPsbMap.size()) {
        throw cms::Exception("Configuration")
        << "\nError: inconsistent length of cable list and input to PSB list.\n"
        << "\n       Can not define the mapping of GCT quadruplets to GT PSBs.\n"
        << std::endl;
    }


    // detailed input configuration for PSB
    std::vector<edm::ParameterSet> psbInput = parSet.getParameter<std::vector<
            edm::ParameterSet> > ("PsbInput");

    // reserve space for L1 GT boards
    m_gtBoardMaps.reserve(boardList.size());


    // fill the maps
    int posVec = 0;

    for (std::vector<std::string>::const_iterator
            it = boardList.begin(); it != boardList.end(); ++it) {

        L1GtBoardType boardType;

        if ( (*it) == "GTFE" ) {
            boardType = GTFE;
        } else if ( (*it) == "FDL" ) {
            boardType = FDL;
        } else if ( (*it) == "PSB" ) {
            boardType = PSB;
        } else if ( (*it) == "GMT" ) {
            boardType = GMT;
        } else if ( (*it) == "TCS" ) {
            boardType = TCS;
        } else if ( (*it) == "TIM" ) {
            boardType = TIM;
        } else {
            throw cms::Exception("Configuration")
            << "\nError: no such board: " << (*it).c_str() << "\n"
            << "\n       Can not define the mapping of the L1 GT boards.     \n"
            << std::endl;
        }

        // construct from board type and board index

        int iBoard = boardIndexVec.at(posVec);
        L1GtBoard gtBoard = L1GtBoard(boardType, iBoard);

        // set the position of board data block
        // in the GT DAQ readout record
        gtBoard.setGtPositionDaqRecord(boardPositionDaqRecord.at(posVec));

        // set the position of board data block
        // in the GT EVM readout record
        gtBoard.setGtPositionEvmRecord(boardPositionEvmRecord.at(posVec));

        // set the bit of board in the GTFE ACTIVE_BOARDS
        // for the GT DAQ readout record
        gtBoard.setGtBitDaqActiveBoards(activeBoardsDaqRecord.at(posVec));

        // set the bit of board in the GTFE ACTIVE_BOARDS
        // for the GT EVM readout record
        gtBoard.setGtBitEvmActiveBoards(activeBoardsEvmRecord.at(posVec));

        // set board slot
        int boardSlot = boardSlotMap.at(posVec);
        gtBoard.setGtBoardSlot(boardSlot);

        // set board hex fragment name in hw record
        gtBoard.setGtBoardHexName(boardHexNameMap.at(posVec));

        // set L1 quadruplet (4x16 bits)(cable) in the PSB input
        // valid for PSB only

        if (boardType == PSB) {

            L1GtPsbQuad psbQuad = Free;
            int posCable = 0;
            int iPsb = 0;
            std::vector<L1GtPsbQuad> quadVec(L1GtBoard::NumberCablesBoard);

            for (std::vector<std::string>::const_iterator
                    cIt = cableList.begin(); cIt != cableList.end(); ++cIt) {


                if ( *cIt == "TechTr" ) {
                    psbQuad = TechTr;
                } else if ( *cIt == "IsoEGQ" ) {
                    psbQuad = IsoEGQ;
                } else if ( *cIt == "NoIsoEGQ" ) {
                    psbQuad = NoIsoEGQ;
                } else if ( *cIt == "CenJetQ" ) {
                    psbQuad = CenJetQ;
                } else if ( *cIt == "ForJetQ" ) {
                    psbQuad = ForJetQ;
                } else if ( *cIt == "TauJetQ" ) {
                    psbQuad = TauJetQ;
                } else if ( *cIt == "ESumsQ" ) {
                    psbQuad = ESumsQ;
                } else if ( *cIt == "JetCountsQ" ) {
                    psbQuad = JetCountsQ;
                } else if ( *cIt == "MQB1" ) {
                    psbQuad = MQB1;
                } else if ( *cIt == "MQB2" ) {
                    psbQuad = MQB2;
                } else if ( *cIt == "MQF3" ) {
                    psbQuad = MQF3;
                } else if ( *cIt == "MQF4" ) {
                    psbQuad = MQF4;
                } else if ( *cIt == "MQB5" ) {
                    psbQuad = MQB5;
                } else if ( *cIt == "MQB6" ) {
                    psbQuad = MQB6;
                } else if ( *cIt == "MQF7" ) {
                    psbQuad = MQF7;
                } else if ( *cIt == "MQF8" ) {
                    psbQuad = MQF8;
                } else if ( *cIt == "MQB9" ) {
                    psbQuad = MQB9;
                } else if ( *cIt == "MQB10" ) {
                    psbQuad = MQB10;
                } else if ( *cIt == "MQF11" ) {
                    psbQuad = MQF11;
                } else if ( *cIt == "MQF12" ) {
                    psbQuad = MQF12;
                } else if ( *cIt == "Free" ) {
                    psbQuad = Free;
                } else if ( *cIt == "HfQ" ) {
                    psbQuad = HfQ;
                } else {
                    // should not arrive here
                    throw cms::Exception("Configuration")
                    << "\nError: no such quadruplet: " << (*cIt).c_str() << "\n"
                    << "\n       Can not define the mapping of quadruplets to the L1 PSB boards.\n"
                    << std::endl;
                }

                int psbIndex = cableToPsbMap.at(posCable);

                if (psbIndex == gtBoard.gtBoardIndex()) {

                    if (iPsb > L1GtBoard::NumberCablesBoard) {
                        throw cms::Exception("Configuration")
                        << "\nError: too many cables for PSB_" << gtBoard.gtBoardIndex()
                        << "\n\n       "
                        << "Can not define the mapping of cables to L1 PSB boards.     \n"
                        << std::endl;

                    }
                    quadVec[iPsb] = psbQuad;
                    iPsb++;
                }
                posCable++;

            }

            gtBoard.setGtQuadInPsb(quadVec);

        }

        if (boardType == PSB) {

            std::map<int, std::vector<L1GtObject> > inputPsbChannels;

            std::vector<std::string> chStrings;
            chStrings.reserve(2); // most channels have 2 objects

            std::vector<L1GtObject>  chObjects;

            for (std::vector<edm::ParameterSet>::const_iterator itPSet =
                    psbInput.begin(); itPSet != psbInput.end(); ++itPSet) {

                //
                int slot = itPSet->getParameter<int> ("Slot");

                if (slot == boardSlot) {
                    chStrings = itPSet->getParameter<std::vector<std::string> > (
                            "Ch0");
                    chObjects = chInputObjects(chStrings);
                    inputPsbChannels[0] = chObjects;
                    chStrings.clear();
                    chObjects.clear();

                    chStrings = itPSet->getParameter<std::vector<std::string> > (
                            "Ch1");
                    chObjects = chInputObjects(chStrings);
                    inputPsbChannels[1] = chObjects;
                    chStrings.clear();
                    chObjects.clear();

                    chStrings = itPSet->getParameter<std::vector<std::string> > (
                            "Ch2");
                    chObjects = chInputObjects(chStrings);
                    inputPsbChannels[2] = chObjects;
                    chStrings.clear();
                    chObjects.clear();

                    chStrings = itPSet->getParameter<std::vector<std::string> > (
                            "Ch3");
                    chObjects = chInputObjects(chStrings);
                    inputPsbChannels[3] = chObjects;
                    chStrings.clear();
                    chObjects.clear();

                    chStrings = itPSet->getParameter<std::vector<std::string> > (
                            "Ch4");
                    chObjects = chInputObjects(chStrings);
                    inputPsbChannels[4] = chObjects;
                    chStrings.clear();
                    chObjects.clear();

                    chStrings = itPSet->getParameter<std::vector<std::string> > (
                            "Ch5");
                    chObjects = chInputObjects(chStrings);
                    inputPsbChannels[5] = chObjects;
                    chStrings.clear();
                    chObjects.clear();

                    chStrings = itPSet->getParameter<std::vector<std::string> > (
                            "Ch6");
                    chObjects = chInputObjects(chStrings);
                    inputPsbChannels[6] = chObjects;
                    chStrings.clear();
                    chObjects.clear();

                    chStrings = itPSet->getParameter<std::vector<std::string> > (
                            "Ch7");
                    chObjects = chInputObjects(chStrings);
                    inputPsbChannels[7] = chObjects;
                    chStrings.clear();
                    chObjects.clear();
                }
            }

            gtBoard.setGtInputPsbChannels(inputPsbChannels);
        }

        // push the board in the vector
        m_gtBoardMaps.push_back(gtBoard);

        // increase the counter
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
std::unique_ptr<L1GtBoardMaps> L1GtBoardMapsTrivialProducer::produceBoardMaps(
    const L1GtBoardMapsRcd& iRecord)
{
    auto pL1GtBoardMaps = std::make_unique<L1GtBoardMaps>();

    pL1GtBoardMaps->setGtBoardMaps(m_gtBoardMaps);

    return pL1GtBoardMaps ;
}
