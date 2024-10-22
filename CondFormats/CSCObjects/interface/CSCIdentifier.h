#ifndef CSCIdentifier_h
#define CSCIdentifier_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include <vector>

class CSCIdentifier {
public:
  CSCIdentifier();
  ~CSCIdentifier();

  struct Item {
    int CSCid;

    COND_SERIALIZABLE;
  };

  std::vector<Item> identifier;

  COND_SERIALIZABLE;
};

#endif
