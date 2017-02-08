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
  IDataItem()
  : m_writeStmt(0),
    m_readStmt(0)
  {}

  virtual std::string getTable() =0;
  

 protected:
  oracle::occi::Statement* m_writeStmt;
  oracle::occi::Statement* m_readStmt;

  inline void checkPrepare() 
    throw(std::runtime_error) 
    {
      if (m_writeStmt == NULL) {
	throw(std::runtime_error("Write statement not prepared"));
      }
    }

  inline void terminateWriteStatement()
    throw(std::runtime_error)
  {
    if (m_writeStmt != NULL) {
      m_conn->terminateStatement(m_writeStmt);
    } else {
      std::cout << "Warning from IDataItem: statement was aleady closed"<< std::endl;
    }
  }


  inline void createReadStatement()
    throw(std::runtime_error)
  {
      m_readStmt=m_conn->createStatement();
  }

  inline void setPrefetchRowCount(int ncount)
    throw(std::runtime_error)
  {
    m_readStmt->setPrefetchRowCount(ncount);
  }

  inline void terminateReadStatement()
    throw(std::runtime_error)
  {
    if (m_readStmt != NULL) {
      m_conn->terminateStatement(m_readStmt);
    } else {
      std::cout << "Warning from IDataItem: statement was aleady closed"<< std::endl;
    }
  }



  // Prepare a statement for writing operations
  virtual void prepareWrite() 
    throw(std::runtime_error) =0;


};

#endif
