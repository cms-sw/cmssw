#ifndef IDATAITEM_H
#define IDATAITEM_H

#include "OnlineDB/EcalCondDB/interface/IDBObject.h"
#include "OnlineDB/EcalCondDB/interface/ITag.h"
#include "OnlineDB/EcalCondDB/interface/IIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

#include <stdexcept>
#include <map>
#include "OnlineDB/Oracle/interface/Oracle.h"

/**
 *   Abstract interface for data in the conditions DB
 */
class IDataItem : public IDBObject {
public:
  IDataItem() : m_writeStmt(nullptr), m_readStmt(nullptr) {}

  virtual std::string getTable() = 0;

protected:
  oracle::occi::Statement* m_writeStmt;
  oracle::occi::Statement* m_readStmt;

  inline void checkPrepare() noexcept(false) {
    if (m_writeStmt == nullptr) {
      throw(std::runtime_error("Write statement not prepared"));
    }
  }

  inline void terminateWriteStatement() noexcept(false) {
    if (m_writeStmt != nullptr) {
      m_conn->terminateStatement(m_writeStmt);
    } else {
      std::cout << "Warning from IDataItem: statement was aleady closed" << std::endl;
    }
  }

  inline void createReadStatement() noexcept(false) { m_readStmt = m_conn->createStatement(); }

  inline void setPrefetchRowCount(int ncount) noexcept(false) { m_readStmt->setPrefetchRowCount(ncount); }

  inline void terminateReadStatement() noexcept(false) {
    if (m_readStmt != nullptr) {
      m_conn->terminateStatement(m_readStmt);
    } else {
      std::cout << "Warning from IDataItem: statement was aleady closed" << std::endl;
    }
  }

  // Prepare a statement for writing operations
  virtual void prepareWrite() noexcept(false) = 0;
};

#endif
