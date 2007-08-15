#ifndef CSCDBGains_h
#define CSCDBGains_h

#include <vector>
#include <map>

class CSCDBGains{
 public:
  CSCDBGains();
  ~CSCDBGains();
  
  struct Item{
    float gain_slope;
    float gain_intercept;
    float gain_chi2;
  };

  //const Item & item(int cscId, int strip) const;

  //typedef std::map< int,std::vector<Item> > GainsMap;
  typedef std::vector<Item> GainsContainer;
  GainsContainer gains;
};

#endif

