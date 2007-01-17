#ifndef IDATAITEM_H
#define IDATAITEM_H

#include "OnlineDB/EcalCondDB/interface/IDBObject.h"
#include "OnlineDB/EcalCondDB/interface/ITag.h"
#include "OnlineDB/EcalCondDB/interface/IIOV.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

#include <stdexcept>
#include <map>
#include "occi.h"

/**
 *   Abstract interface for data in the conditions DB
 */
class IDataItem : public IDBObject {
 public:
  virtual std::string getTable() =0;

 protected:
  oracle::occi::Statement* m_writeStmt;

  inline void checkPrepare() 
    throw(std::runtime_error) 
    {
      if (m_writeStmt == NULL) {
	throw(std::runtime_error("Write statement not prepared"));
      }
    }

  // Prepare a statement for writing operations
  virtual void prepareWrite() 
    throw(std::runtime_error) =0;

};

#endif
