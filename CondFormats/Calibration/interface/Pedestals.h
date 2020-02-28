#ifndef PEDESTALS_H
#define PEDESTALS_H
#include "CondFormats/Serialization/interface/Serializable.h"

#include <vector>
class Pedestals {
public:
  struct Item {
    Item() {}
    ~Item() {}
    float m_mean;
    float m_variance;

    COND_SERIALIZABLE;
  };
  Pedestals();
  virtual ~Pedestals() {}
  typedef std::vector<Item>::const_iterator ItemIterator;
  std::vector<Item> m_pedestals;

  COND_SERIALIZABLE;
};
#endif
