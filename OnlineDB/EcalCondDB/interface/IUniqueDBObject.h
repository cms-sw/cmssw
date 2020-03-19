#ifndef IUNIQUEDBOBJECT_H
#define IUNIQUEDBOBJECT_H

#include <stdexcept>

#include "OnlineDB/EcalCondDB/interface/IDBObject.h"

/**
 *   Abstract base class for an object with a single unique ID
 */
class IUniqueDBObject : public IDBObject {
public:
  virtual int fetchID() noexcept(false) = 0;
  virtual void setByID(int id) noexcept(false) = 0;

protected:
  // ID from the database
  int m_ID;
};

#endif
