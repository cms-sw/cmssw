#ifndef CondFormats_EcalObjects_EcalXtalGroupId_H
#define CondFormats_EcalObjects_EcalXtalGroupId_H

#include "CondFormats/Serialization/interface/Serializable.h"
/**
 * Author: Shahram Rahatlou, University of Rome & INFN
 * Created: 22 Feb 2006
 * $Id: EcalXtalGroupId.h,v 1.3 2006/02/23 16:56:34 rahatlou Exp $
 **/

class EcalXtalGroupId {
public:
  EcalXtalGroupId() : id_(0) {}
  EcalXtalGroupId(const unsigned int& id) : id_(id) {}

  bool operator>(const EcalXtalGroupId& rhs) const { return (id_ > rhs.id()); }
  bool operator>=(const EcalXtalGroupId& rhs) const { return (id_ >= rhs.id()); }
  bool operator==(const EcalXtalGroupId& rhs) const { return (id_ == rhs.id()); }
  bool operator<(const EcalXtalGroupId& rhs) const { return (id_ < rhs.id()); }
  bool operator<=(const EcalXtalGroupId& rhs) const { return (id_ <= rhs.id()); }

  const unsigned int id() const { return id_; }

private:
  unsigned int id_;

  COND_SERIALIZABLE;
};
#endif
