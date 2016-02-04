#ifndef IUNIQUEDBOBJECT_H
#define IUNIQUEDBOBJECT_H

#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDBObject.h"

/**
 *   Abstract base class for an object with a single unique ID
 */
class IUniqueDBObject : public IDBObject {
 public:
  virtual int fetchID() throw(std::runtime_error) =0;
  virtual void setByID(int id) throw(std::runtime_error) =0;

 protected:
  // ID from the database
  int m_ID;
};

#endif
