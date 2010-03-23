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
  };
    
  CentralityTable(){}
  std::vector<CBin> m_table;
};

