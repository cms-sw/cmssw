#ifndef CondFormats_ESObjects_ESStripGroupId_H
#define CondFormats_ESObjects_ESStripGroupId_H

#include "CondFormats/Serialization/interface/Serializable.h"

class ESStripGroupId {
public:
  ESStripGroupId() : id_(0) {}
  ESStripGroupId(const unsigned int& id) : id_(id) {}

  bool operator>(const ESStripGroupId& rhs) const { return (id_ > rhs.id()); }
  bool operator>=(const ESStripGroupId& rhs) const { return (id_ >= rhs.id()); }
  bool operator==(const ESStripGroupId& rhs) const { return (id_ == rhs.id()); }
  bool operator<(const ESStripGroupId& rhs) const { return (id_ < rhs.id()); }
  bool operator<=(const ESStripGroupId& rhs) const { return (id_ <= rhs.id()); }

  const unsigned int id() const { return id_; }

private:
  unsigned int id_;

  COND_SERIALIZABLE;
};
#endif
