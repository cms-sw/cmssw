//-------------------------------------------------
//
//   class L1MuGMTInputEvent
//
//   Description:
//
//
//   Author :
//   Tobias Noebauer              HEPHY Vienna
//
//   Migrated to CMSSW:
//   I. Mikulec
//
//--------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTInputEvent.h"

//---------------
// C++ Headers --
//---------------
#include <stdexcept>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTMatrix.h"
//#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTConfig.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"

//----------------
// Constructors --
//----------------
L1MuGMTInputEvent::L1MuGMTInputEvent()
    : m_runnr(0L),
      m_evtnr(0L),
      m_mip_bits(14, 18, false),
      m_iso_bits(14, 18, true)  //this is more useful when reading a standalone input file
                                //since "not-quiet" bits are stored there
{
  std::vector<L1MuRegionalCand> empty_vec;
  m_inputmuons["INC"] = empty_vec;
  m_inputmuons["IND"] = empty_vec;
  m_inputmuons["INB"] = empty_vec;
  m_inputmuons["INF"] = empty_vec;

  //  m_inputmuons["INC"].reserve(L1MuGMTConfig::MAXCSC);
  //  m_inputmuons["IND"].reserve(L1MuGMTConfig::MAXDTBX);
  //  m_inputmuons["INB"].reserve(L1MuGMTConfig::MAXRPC);
  //  m_inputmuons["INF"].reserve(L1MuGMTConfig::MAXRPC);
  m_inputmuons["INC"].reserve(4);
  m_inputmuons["IND"].reserve(4);
  m_inputmuons["INB"].reserve(4);
  m_inputmuons["INF"].reserve(4);
}

//--------------
// Destructor --
//--------------
L1MuGMTInputEvent::~L1MuGMTInputEvent() {}

//--------------
// Operations --
//--------------
void L1MuGMTInputEvent::addInputMuon(const std::string chipid, const L1MuRegionalCand& inMu) {
  if (m_inputmuons.count(chipid) == 0)
    throw std::runtime_error("L1MuGMTInputEvent::addInputMuon: invalid chipid:" + chipid);
  m_inputmuons[chipid].push_back(inMu);
}

void L1MuGMTInputEvent::reset() {
  m_runnr = 0L;
  m_evtnr = 0L;

  std::map<std::string, std::vector<L1MuRegionalCand> >::iterator it = m_inputmuons.begin();
  for (; it != m_inputmuons.end(); it++) {
    it->second.clear();
  }

  m_mip_bits.reset(false);
  m_iso_bits.reset(true);  //see CTOR for info on this
}

const L1MuRegionalCand* L1MuGMTInputEvent::getInputMuon(std::string chipid, unsigned index) const {
  if (m_inputmuons.count(chipid) == 0)
    throw std::runtime_error("L1GMTInputEvent::getInputMuon: invalid chipid:" + chipid);

  if (index >= m_inputmuons.find(chipid)->second.size())
    return nullptr;
  return &(m_inputmuons.find(chipid)->second.at(index));
}
