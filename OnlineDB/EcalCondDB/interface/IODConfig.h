#ifndef IODCONFIG_H
#define IODCONFIG_H

#include <stdexcept>
#include "OnlineDB/Oracle/interface/Oracle.h"
#include "OnlineDB/EcalCondDB/interface/IDBObject.h"

/**
 *   Abstract interface for data in the conditions DB
 */

class IODConfig : public IDBObject {

 public:

  std::string   m_config_tag;

  virtual std::string getTable() =0;

  inline void setConfigTag(std::string x) {m_config_tag=x;}
  inline std::string getConfigTag() {return m_config_tag;}
 

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
      cout << "Warning from IDataItem: statement was aleady closed"<< endl;
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
      cout << "Warning from IDataItem: statement was aleady closed"<< endl;
    }
  }



  // Prepare a statement for writing operations
  virtual void prepareWrite() throw(std::runtime_error) =0;

  //  virtual void writeDB() throw(std::runtime_error) ;




};

#endif


