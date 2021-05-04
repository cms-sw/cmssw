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

// this class header
#include "CondFormats/L1TObjects/interface/L1GtBoardMaps.h"

// system include files
#include <iostream>
#include <iomanip>

// user include files

// forward declarations

// constructor
L1GtBoardMaps::L1GtBoardMaps() {
  // empty
}

// destructor
L1GtBoardMaps::~L1GtBoardMaps() {
  // empty
}

// set / print the L1 GT board map
void L1GtBoardMaps::setGtBoardMaps(const std::vector<L1GtBoard>& gtBoardMapsValue) { m_gtBoardMaps = gtBoardMapsValue; }

void L1GtBoardMaps::print(std::ostream& myCout) const {
  myCout << "\nL1 GT board map" << std::endl;

  myCout << "  Size: " << m_gtBoardMaps.size() << " boards in L1 GT." << std::endl;

  for (std::vector<L1GtBoard>::const_iterator cIt = m_gtBoardMaps.begin(); cIt != m_gtBoardMaps.end(); ++cIt) {
    cIt->print(myCout);
    myCout << std::endl;
  }

  myCout << std::endl;
}

// print L1 GT DAQ record map
void L1GtBoardMaps::printGtDaqRecordMap(std::ostream& myCout) const {
  myCout << "\nL1 GT DAQ record map" << std::endl;

  int nrBoards = 0;
  int posRec = -1;
  uint16_t boardId = 0;
  std::string boardName;

  for (std::vector<L1GtBoard>::const_iterator cIt = m_gtBoardMaps.begin(); cIt != m_gtBoardMaps.end(); ++cIt) {
    posRec = cIt->gtPositionDaqRecord();
    boardId = cIt->gtBoardId();
    boardName = cIt->gtBoardName();

    if (posRec >= 0) {
      myCout << "  " << boardName << "  " << std::hex << boardId << std::dec << " at position " << posRec << std::endl;

      nrBoards++;
    }
  }

  myCout << "\n  Size: " << nrBoards << " boards in the record" << std::endl;
  myCout << "  Header and trailer are automatically added to the hardware record.\n" << std::endl;

  myCout << std::endl;
}

// print L1 GT EVM record map
void L1GtBoardMaps::printGtEvmRecordMap(std::ostream& myCout) const {
  myCout << "\nL1 GT EVM record map" << std::endl;

  int nrBoards = 0;
  int posRec = -1;
  uint16_t boardId = 0;
  std::string boardName;

  for (std::vector<L1GtBoard>::const_iterator cIt = m_gtBoardMaps.begin(); cIt != m_gtBoardMaps.end(); ++cIt) {
    posRec = cIt->gtPositionEvmRecord();
    boardId = cIt->gtBoardId();
    boardName = cIt->gtBoardName();

    if (posRec >= 0) {
      myCout << "  " << boardName << "  " << std::hex << boardId << std::dec << " at position " << posRec << std::endl;

      nrBoards++;
    }
  }

  myCout << "\n  Size: " << nrBoards << " boards in the record" << std::endl;
  myCout << "  Header and trailer are automatically added to the hardware record.\n" << std::endl;

  myCout << std::endl;
}

// print L1 GT active boards map for DAQ record
void L1GtBoardMaps::printGtDaqActiveBoardsMap(std::ostream& myCout) const {
  myCout << "\nL1 GT DAQ \"active boards\" record map" << std::endl;

  int nrBoards = 0;
  int posRec = -1;
  uint16_t boardId = 0;
  std::string boardName;

  for (std::vector<L1GtBoard>::const_iterator cIt = m_gtBoardMaps.begin(); cIt != m_gtBoardMaps.end(); ++cIt) {
    posRec = cIt->gtBitDaqActiveBoards();
    boardId = cIt->gtBoardId();
    boardName = cIt->gtBoardName();

    if (posRec >= 0) {
      myCout << "  " << boardName << "  " << std::hex << boardId << std::dec << " at bit " << posRec << std::endl;

      nrBoards++;
    }
  }

  myCout << "\n  Size: " << nrBoards << " boards in the record" << std::endl;
  myCout << std::endl;

  myCout << std::endl;
}

// print L1 GT active boards map for EVM record
void L1GtBoardMaps::printGtEvmActiveBoardsMap(std::ostream& myCout) const {
  myCout << "\nL1 GT EVM \"active boards\" record map" << std::endl;

  int nrBoards = 0;
  int posRec = -1;
  uint16_t boardId = 0;
  std::string boardName;

  for (std::vector<L1GtBoard>::const_iterator cIt = m_gtBoardMaps.begin(); cIt != m_gtBoardMaps.end(); ++cIt) {
    posRec = cIt->gtBitEvmActiveBoards();
    boardId = cIt->gtBoardId();
    boardName = cIt->gtBoardName();

    if (posRec >= 0) {
      myCout << "  " << boardName << "  " << std::hex << boardId << std::dec << " at bit " << posRec << std::endl;

      nrBoards++;
    }
  }

  myCout << "\n  Size: " << nrBoards << " boards in the record" << std::endl;
  myCout << std::endl;

  myCout << std::endl;
}

// print L1 GT board - slot map
void L1GtBoardMaps::printGtBoardSlotMap(std::ostream& myCout) const {
  myCout << "\nL1 GT board - slot map" << std::endl;

  int nrBoards = 0;
  int posRec = -1;
  uint16_t boardId = 0;
  std::string boardName;

  for (std::vector<L1GtBoard>::const_iterator cIt = m_gtBoardMaps.begin(); cIt != m_gtBoardMaps.end(); ++cIt) {
    posRec = cIt->gtBoardSlot();
    boardId = cIt->gtBoardId();
    boardName = cIt->gtBoardName();

    if (posRec >= 0) {
      myCout << "  " << boardName << "  " << std::hex << boardId << std::dec << " in slot " << posRec << std::endl;

      nrBoards++;
    }
  }

  myCout << "\n  Size: " << nrBoards << " boards in the slot map" << std::endl;
  myCout << std::endl;

  myCout << std::endl;
}

// print L1 GT board name in hw record map
void L1GtBoardMaps::printGtBoardHexNameMap(std::ostream& myCout) const {
  myCout << "\nL1 GT board names in hw record map" << std::endl;

  int nrBoards = 0;
  int posRec = -1;
  uint16_t boardId = 0;
  std::string boardName;

  for (std::vector<L1GtBoard>::const_iterator cIt = m_gtBoardMaps.begin(); cIt != m_gtBoardMaps.end(); ++cIt) {
    posRec = cIt->gtBoardHexName();
    boardId = cIt->gtBoardId();
    boardName = cIt->gtBoardName();

    if (posRec >= 0) {
      myCout << "  " << boardName << "  " << std::hex << boardId << std::dec << " has HexName " << std::hex << posRec
             << std::dec << std::endl;

      nrBoards++;
    }
  }

  myCout << "\n  Size: " << nrBoards << " boards in the record" << std::endl;
  myCout << std::endl;

  myCout << std::endl;
}

// print L1 quadruplet (4x16 bits)(cable) to PSB input map
void L1GtBoardMaps::printGtQuadToPsbMap(std::ostream& myCout) const {
  myCout << "\nL1 GT \"cables to PSB\" input map (4x16 bits per cable) - DEPRECATED" << std::endl;

  int nrBoards = 0;
  int nrCable = 0;

  for (std::vector<L1GtBoard>::const_iterator cIt = m_gtBoardMaps.begin(); cIt != m_gtBoardMaps.end(); ++cIt) {
    if (cIt->gtBoardType() == PSB) {
      myCout << "\n  PSB_" << cIt->gtBoardIndex() << "\n      ";

      nrBoards++;

      std::vector<L1GtPsbQuad> quadInPsb = cIt->gtQuadInPsb();
      std::string objType;

      for (std::vector<L1GtPsbQuad>::const_iterator cItQuad = quadInPsb.begin(); cItQuad != quadInPsb.end();
           ++cItQuad) {
        nrCable++;

        if (*cItQuad == TechTr) {
          objType = "TechTr";
        } else if (*cItQuad == IsoEGQ) {
          objType = "IsoEGQ";
        } else if (*cItQuad == NoIsoEGQ) {
          objType = "NoIsoEGQ";
        } else if (*cItQuad == CenJetQ) {
          objType = "CenJetQ";
        } else if (*cItQuad == ForJetQ) {
          objType = "ForJetQ";
        } else if (*cItQuad == TauJetQ) {
          objType = "TauJetQ";
        } else if (*cItQuad == ESumsQ) {
          objType = "ESumsQ";
        } else if (*cItQuad == JetCountsQ) {
          objType = "JetCountsQ";
        } else if (*cItQuad == MQB1) {
          objType = "MQB1";
        } else if (*cItQuad == MQB2) {
          objType = "MQB2";
        } else if (*cItQuad == MQF3) {
          objType = "MQF3";
        } else if (*cItQuad == MQF4) {
          objType = "MQF4";
        } else if (*cItQuad == MQB5) {
          objType = "MQB5";
        } else if (*cItQuad == MQB6) {
          objType = "MQB6";
        } else if (*cItQuad == MQF7) {
          objType = "MQF7";
        } else if (*cItQuad == MQF8) {
          objType = "MQF8";
        } else if (*cItQuad == MQB9) {
          objType = "MQB9";
        } else if (*cItQuad == MQB10) {
          objType = "MQB10";
        } else if (*cItQuad == MQF11) {
          objType = "MQF11";
        } else if (*cItQuad == MQF12) {
          objType = "MQF12";
        } else if (*cItQuad == Free) {
          objType = "Free";
        } else if (*cItQuad == HfQ) {
          objType = "HfQ";
        } else {
          myCout << "\n\nError: no such member " << (*cItQuad) << " in enum L1GtPsbQuad\n\n" << std::endl;
          objType = "ERROR";
        }

        myCout << objType << " ";
      }
    }
  }

  myCout << "\n\n  Size: " << nrCable << " cables for " << nrBoards << " PSB boards" << std::endl;

  myCout << std::endl;
}

// output stream operator
std::ostream& operator<<(std::ostream& os, const L1GtBoardMaps& result) {
  result.print(os);
  return os;
}
