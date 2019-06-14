#ifndef BLOBPEDESTALS_H
#define BLOBPEDESTALS_H
#include "CondFormats/Serialization/interface/Serializable.h"

#include <vector>
class BlobPedestals {
public:
  BlobPedestals();
  virtual ~BlobPedestals();
  std::vector<unsigned int> m_pedestals;

  COND_SERIALIZABLE;
};
#endif
