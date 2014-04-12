#ifndef CSCDBPedestals_h
#define CSCDBPedestals_h

#include <iosfwd>
#include <vector>

class CSCDBPedestals{
 public:
  CSCDBPedestals(){}
  ~CSCDBPedestals(){}

  struct Item{
    short int ped;
    short int rms;
  };
  int factor_ped;
  int factor_rms;

  enum factors{FPED=10, FRMS=1000};

  typedef std::vector<Item> PedestalContainer;
  PedestalContainer pedestals;

  const Item & item(int index) const { return pedestals[index]; }
  short int pedestal( int index ) const { return pedestals[index].ped; }
  int scale_ped() const { return factor_ped; }
  short int pedestal_rms( int index ) const { return pedestals[index].rms; }
  int scale_rms() const { return factor_rms; }
};

std::ostream & operator<<(std::ostream & os, const CSCDBPedestals & cscdb);

#endif
