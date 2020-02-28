//---------------------------------------------
//
//   \class L1MuGMTReadoutCollection
//
//   Description:
//
//
//
//   Author :
//   Hannes Sakulin                  HEPHY Vienna
//   Ivan Mikulec                    HEPHY Vienna
//
//--------------------------------------------------
#ifndef DataFormatsL1GlobalMuonTrigger_L1MuGMTReadoutCollection_h
#define DataFormatsL1GlobalMuonTrigger_L1MuGMTReadoutCollection_h

//---------------
// C++ Headers --
//---------------

#include <vector>
#include <map>
#include <iostream>

//----------------------
// Base Class Headers --
//----------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutRecord.h"

//---------------------
//-- Class Interface --
//---------------------

class L1MuGMTReadoutCollection {
public:
  L1MuGMTReadoutCollection(){};
  L1MuGMTReadoutCollection(int nbx) { m_Records.reserve(nbx); };

  virtual ~L1MuGMTReadoutCollection(){};

  void reset() {
    for (unsigned int i = 0; i < m_Records.size(); i++)
      m_Records[i].reset();
  };

  // get record vector
  std::vector<L1MuGMTReadoutRecord> const& getRecords() const { return m_Records; };

  // get record for a given bx
  L1MuGMTReadoutRecord const& getRecord(int bx = 0) const {
    std::vector<L1MuGMTReadoutRecord>::const_iterator iter;
    for (iter = m_Records.begin(); iter != m_Records.end(); iter++) {
      if ((*iter).getBxCounter() == bx)
        return (*iter);
    }
    return getDefaultFor(bx);
  };

  // add record
  void addRecord(L1MuGMTReadoutRecord const& rec) { m_Records.push_back(rec); };

private:
  static L1MuGMTReadoutRecord const& getDefaultFor(int bx);

  std::vector<L1MuGMTReadoutRecord> m_Records;
};

#endif  // DataFormatsL1GlobalMuonTrigger_L1MuGMTReadoutCollection_h
