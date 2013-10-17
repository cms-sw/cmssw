#ifndef __CentralityTable_h__
#define __CentralityTable_h__

#include "CondFormats/Serialization/interface/Serializable.h"

#include <vector>
class CentralityTable {
  
  public:

  struct BinValues{
    float mean;
    float var;
  
  COND_SERIALIZABLE;
};

  struct CBin {
    float bin_edge;
    BinValues n_part;
    BinValues n_coll;
    BinValues n_hard;
    BinValues b;

    BinValues eccRP;
    BinValues ecc2;
    BinValues ecc3;
    BinValues ecc4;
    BinValues ecc5;

    BinValues S;

    BinValues var0;
    BinValues var1;
    BinValues var2;
  
  COND_SERIALIZABLE;
};
    
  CentralityTable(){}
  std::vector<CBin> m_table;

  COND_SERIALIZABLE;
};

#endif

