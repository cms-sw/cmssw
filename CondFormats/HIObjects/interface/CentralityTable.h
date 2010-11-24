#include <vector>
class CentralityTable {
  
  public:

  struct BinValues{
    float mean;
    float var;
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
  };
    
  CentralityTable(){}
  std::vector<CBin> m_table;
};

