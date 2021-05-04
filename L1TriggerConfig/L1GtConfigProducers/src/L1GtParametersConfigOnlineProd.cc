/**
 * \class L1GtParametersConfigOnlineProd
 *
 *
 * Description: online producer for L1GtParameters.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 *
 *
 */

// this class header
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtParametersConfigOnlineProd.h"

// system include files
#include "boost/lexical_cast.hpp"

// user include files
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// constructor
L1GtParametersConfigOnlineProd::L1GtParametersConfigOnlineProd(const edm::ParameterSet& parSet)
    : L1ConfigOnlineProdBase<L1GtParametersRcd, L1GtParameters>(parSet) {
  // empty
}

// destructor
L1GtParametersConfigOnlineProd::~L1GtParametersConfigOnlineProd() {
  // empty
}

// public methods

std::unique_ptr<L1GtParameters> L1GtParametersConfigOnlineProd::newObject(const std::string& objectKey) {
  auto pL1GtParameters = std::make_unique<L1GtParameters>();

  // l1GtParameters: parameters in table GTFE_SETUP_FK

  const std::string gtSchema = "CMS_GT";

  // SQL query:
  //   SELECT EVM_INCLUDE_TCS,
  //          EVM_INCLUDE_FDL,
  //          DAQ_INCLUDE_FDL,
  //          DAQ_INCLUDE_PSB0,
  //          DAQ_INCLUDE_PSB1,
  //          DAQ_INCLUDE_PSB2,
  //          DAQ_INCLUDE_PSB3,
  //          DAQ_INCLUDE_PSB4,
  //          DAQ_INCLUDE_PSB5,
  //          DAQ_INCLUDE_PSB6,
  //          DAQ_INCLUDE_GMT,
  //          DAQ_INCLUDE_TIM,
  //          DAQ_NB_BC_PER_EVENT_FDL,
  //          BST_DATA_NB_BYTES
  //   FROM CMS_GT.GT_GTFE_SETUP
  //   WHERE GT_GTFE_SETUP.ID = objectKey

  std::vector<std::string> columns;
  columns.push_back("EVM_INCLUDE_TCS");
  columns.push_back("EVM_INCLUDE_FDL");
  columns.push_back("DAQ_INCLUDE_FDL");
  columns.push_back("DAQ_INCLUDE_PSB0");
  columns.push_back("DAQ_INCLUDE_PSB1");
  columns.push_back("DAQ_INCLUDE_PSB2");
  columns.push_back("DAQ_INCLUDE_PSB3");
  columns.push_back("DAQ_INCLUDE_PSB4");
  columns.push_back("DAQ_INCLUDE_PSB5");
  columns.push_back("DAQ_INCLUDE_PSB6");
  columns.push_back("DAQ_INCLUDE_GMT");
  columns.push_back("DAQ_INCLUDE_TIM");
  columns.push_back("DAQ_NB_BC_PER_EVENT_FDL");
  columns.push_back("DAQ_NB_BC_PER_EVENT_PSB0");
  columns.push_back("DAQ_NB_BC_PER_EVENT_PSB1");
  columns.push_back("DAQ_NB_BC_PER_EVENT_PSB2");
  columns.push_back("DAQ_NB_BC_PER_EVENT_PSB3");
  columns.push_back("DAQ_NB_BC_PER_EVENT_PSB4");
  columns.push_back("DAQ_NB_BC_PER_EVENT_PSB5");
  columns.push_back("DAQ_NB_BC_PER_EVENT_PSB6");
  columns.push_back("DAQ_NB_BC_PER_EVENT_GMT");
  columns.push_back("DAQ_NB_BC_PER_EVENT_TIM");
  columns.push_back("BST_DATA_NB_BYTES");

  l1t::OMDSReader::QueryResults results = m_omdsReader.basicQuery(
      columns, gtSchema, "GT_GTFE_SETUP", "GT_GTFE_SETUP.ID", m_omdsReader.singleAttribute(objectKey));

  // check if query was successful
  if (results.queryFailed()) {
    edm::LogError("L1-O2O") << "Problem with L1GtParameters key:" << objectKey;
    return pL1GtParameters;
  }

  bool activeBoardsEvmTCS = false;
  results.fillVariable("EVM_INCLUDE_TCS", activeBoardsEvmTCS);

  bool activeBoardsEvmFDL = false;
  results.fillVariable("EVM_INCLUDE_FDL", activeBoardsEvmFDL);

  bool activeBoardsDaqFDL = false;
  results.fillVariable("DAQ_INCLUDE_FDL", activeBoardsDaqFDL);

  bool activeBoardsDaqPSB0 = false;
  results.fillVariable("DAQ_INCLUDE_PSB0", activeBoardsDaqPSB0);

  bool activeBoardsDaqPSB1 = false;
  results.fillVariable("DAQ_INCLUDE_PSB1", activeBoardsDaqPSB1);

  bool activeBoardsDaqPSB2 = false;
  results.fillVariable("DAQ_INCLUDE_PSB2", activeBoardsDaqPSB2);

  bool activeBoardsDaqPSB3 = false;
  results.fillVariable("DAQ_INCLUDE_PSB3", activeBoardsDaqPSB3);

  bool activeBoardsDaqPSB4;
  results.fillVariable("DAQ_INCLUDE_PSB4", activeBoardsDaqPSB4);

  bool activeBoardsDaqPSB5 = false;
  results.fillVariable("DAQ_INCLUDE_PSB5", activeBoardsDaqPSB5);

  bool activeBoardsDaqPSB6;
  results.fillVariable("DAQ_INCLUDE_PSB6", activeBoardsDaqPSB6);

  bool activeBoardsDaqGMT;
  results.fillVariable("DAQ_INCLUDE_GMT", activeBoardsDaqGMT);

  bool activeBoardsDaqTIM = false;
  results.fillVariable("DAQ_INCLUDE_TIM", activeBoardsDaqTIM);

  std::string totalBxInEventStr;
  results.fillVariable("DAQ_NB_BC_PER_EVENT_FDL", totalBxInEventStr);

  //
  std::string daqNrBxBoardStrFDL;
  results.fillVariable("DAQ_NB_BC_PER_EVENT_FDL", daqNrBxBoardStrFDL);

  std::string daqNrBxBoardStrPSB0;
  results.fillVariable("DAQ_NB_BC_PER_EVENT_PSB0", daqNrBxBoardStrPSB0);

  std::string daqNrBxBoardStrPSB1;
  results.fillVariable("DAQ_NB_BC_PER_EVENT_PSB1", daqNrBxBoardStrPSB1);

  std::string daqNrBxBoardStrPSB2;
  results.fillVariable("DAQ_NB_BC_PER_EVENT_PSB2", daqNrBxBoardStrPSB2);

  std::string daqNrBxBoardStrPSB3;
  results.fillVariable("DAQ_NB_BC_PER_EVENT_PSB3", daqNrBxBoardStrPSB3);

  std::string daqNrBxBoardStrPSB4;
  results.fillVariable("DAQ_NB_BC_PER_EVENT_PSB4", daqNrBxBoardStrPSB4);

  std::string daqNrBxBoardStrPSB5;
  results.fillVariable("DAQ_NB_BC_PER_EVENT_PSB5", daqNrBxBoardStrPSB5);

  std::string daqNrBxBoardStrPSB6;
  results.fillVariable("DAQ_NB_BC_PER_EVENT_PSB6", daqNrBxBoardStrPSB6);

  std::string daqNrBxBoardStrGMT;
  results.fillVariable("DAQ_NB_BC_PER_EVENT_GMT", daqNrBxBoardStrGMT);

  std::string daqNrBxBoardStrTIM;
  results.fillVariable("DAQ_NB_BC_PER_EVENT_TIM", daqNrBxBoardStrTIM);

  std::string bstLengthBytesStr;
  results.fillVariable("BST_DATA_NB_BYTES", bstLengthBytesStr);

  // fill the record

  // total Bx's in the event
  int totalBxInEventVal = boost::lexical_cast<int>(totalBxInEventStr);
  pL1GtParameters->setGtTotalBxInEvent(totalBxInEventVal);

  // FIXME: need board maps in DB, with active bit number...
  //        now, they are hardwired

  // get the mapping of boards to active bits

  // active boards in the L1 DAQ record & number of BXs per board
  uint16_t daqActiveBoardsVal = 0;

  int daqActiveBoardsLength = 16;  // ...hard...
  std::vector<int> daqNrBxBoard(daqActiveBoardsLength, 0);

  int iActiveBit = 0;
  if (activeBoardsDaqFDL) {
    daqActiveBoardsVal = daqActiveBoardsVal | (1 << iActiveBit);
  }
  daqNrBxBoard.at(iActiveBit) = boost::lexical_cast<int>(daqNrBxBoardStrFDL);

  iActiveBit = 1;
  if (activeBoardsDaqPSB0) {
    daqActiveBoardsVal = daqActiveBoardsVal | (1 << iActiveBit);
  }
  daqNrBxBoard.at(iActiveBit) = boost::lexical_cast<int>(daqNrBxBoardStrPSB0);

  iActiveBit = 2;
  if (activeBoardsDaqPSB1) {
    daqActiveBoardsVal = daqActiveBoardsVal | (1 << iActiveBit);
  }
  daqNrBxBoard.at(iActiveBit) = boost::lexical_cast<int>(daqNrBxBoardStrPSB1);

  iActiveBit = 3;
  if (activeBoardsDaqPSB2) {
    daqActiveBoardsVal = daqActiveBoardsVal | (1 << iActiveBit);
  }
  daqNrBxBoard.at(iActiveBit) = boost::lexical_cast<int>(daqNrBxBoardStrPSB2);

  iActiveBit = 4;
  if (activeBoardsDaqPSB3) {
    daqActiveBoardsVal = daqActiveBoardsVal | (1 << iActiveBit);
  }
  daqNrBxBoard.at(iActiveBit) = boost::lexical_cast<int>(daqNrBxBoardStrPSB3);

  iActiveBit = 5;
  if (activeBoardsDaqPSB4) {
    daqActiveBoardsVal = daqActiveBoardsVal | (1 << iActiveBit);
  }
  daqNrBxBoard.at(iActiveBit) = boost::lexical_cast<int>(daqNrBxBoardStrPSB4);

  iActiveBit = 6;
  if (activeBoardsDaqPSB5) {
    daqActiveBoardsVal = daqActiveBoardsVal | (1 << iActiveBit);
  }
  daqNrBxBoard.at(iActiveBit) = boost::lexical_cast<int>(daqNrBxBoardStrPSB5);

  iActiveBit = 7;
  if (activeBoardsDaqPSB6) {
    daqActiveBoardsVal = daqActiveBoardsVal | (1 << iActiveBit);
  }
  daqNrBxBoard.at(iActiveBit) = boost::lexical_cast<int>(daqNrBxBoardStrPSB6);

  iActiveBit = 8;
  if (activeBoardsDaqGMT) {
    daqActiveBoardsVal = daqActiveBoardsVal | (1 << iActiveBit);
  }
  daqNrBxBoard.at(iActiveBit) = boost::lexical_cast<int>(daqNrBxBoardStrGMT);

  // FIXME fix TIM active bit
  //iActiveBit = 9;
  //if (activeBoardsDaqTIM) {
  //    daqActiveBoardsVal = daqActiveBoardsVal | (1 << iActiveBit);
  //}
  //daqNrBxBoard.at(iActiveBit) = boost::lexical_cast<int>(daqNrBxBoardStrTIM);

  // active boards in the L1 EVM record
  uint16_t evmActiveBoardsVal = 0;

  int evmActiveBoardsLength = 16;  // ...hard...
  std::vector<int> evmNrBxBoard(evmActiveBoardsLength, 0);

  iActiveBit = 0;
  if (activeBoardsEvmTCS) {
    evmActiveBoardsVal = evmActiveBoardsVal | (1 << iActiveBit);
  }
  // always 1 TCS payload - hardcoded
  evmNrBxBoard.at(iActiveBit) = 1;

  iActiveBit = 1;
  if (activeBoardsEvmFDL) {
    evmActiveBoardsVal = evmActiveBoardsVal | (1 << iActiveBit);
  }
  // FDL must have the same length in EVM and DAQ
  evmNrBxBoard.at(iActiveBit) = boost::lexical_cast<int>(daqNrBxBoardStrFDL);

  //
  pL1GtParameters->setGtDaqActiveBoards(daqActiveBoardsVal);
  pL1GtParameters->setGtEvmActiveBoards(evmActiveBoardsVal);
  pL1GtParameters->setGtDaqNrBxBoard(daqNrBxBoard);
  pL1GtParameters->setGtEvmNrBxBoard(evmNrBxBoard);

  //
  unsigned int bstLengthBytesVal = boost::lexical_cast<unsigned int>(bstLengthBytesStr);
  pL1GtParameters->setGtBstLengthBytes(bstLengthBytesVal);

  if (edm::isDebugEnabled()) {
    std::ostringstream myCoutStream;
    pL1GtParameters->print(myCoutStream);
    LogTrace("L1GtParametersConfigOnlineProd") << "\nThe following L1GtParameters record was read from OMDS: \n"
                                               << myCoutStream.str() << "\n"
                                               << std::endl;
  }

  return pL1GtParameters;
}

DEFINE_FWK_EVENTSETUP_MODULE(L1GtParametersConfigOnlineProd);
