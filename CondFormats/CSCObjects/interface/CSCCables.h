#ifndef CSCCables_h
#define CSCCables_h

#include <vector>

class CSCCables{
 public:
  CSCCables();
  ~CSCCables();
  
  struct Cables{
    std::string chamber_label;
    int cfeb_length;
    std::string cfeb_rev;
    int alct_length;
    std::string alct_rev;
    int cfeb_tmb_skew_delay;
    int cfeb_timing_corr;
  };

  typedef std::vector<Cables> CablesContainer;

  CablesContainer cables;
};

#endif
