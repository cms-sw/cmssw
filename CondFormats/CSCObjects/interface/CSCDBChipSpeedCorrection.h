#ifndef CSCDBChipSpeedCorrection_h
#define CSCDBChipSpeedCorrection_h

#include <iosfwd>
#include <vector>

class CSCDBChipSpeedCorrection{
 public:
  CSCDBChipSpeedCorrection(){}
  ~CSCDBChipSpeedCorrection(){}

  struct Item{
    short int speedCorr;
  };
  int factor_speedCorr;

  enum factors{FCORR=100};

  typedef std::vector<Item> ChipSpeedContainer;
  ChipSpeedContainer chipSpeedCorr;

  const Item & item( int index) const { return chipSpeedCorr[index]; }
  short int value( int index ) const { return chipSpeedCorr[index].speedCorr; }
  int scale() const { return factor_speedCorr; }
};

std::ostream & operator<<(std::ostream & os, const CSCDBChipSpeedCorrection & cscdb);

#endif
