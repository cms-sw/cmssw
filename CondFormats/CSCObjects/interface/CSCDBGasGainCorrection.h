#ifndef CSCDBGasGainCorrection_h
#define CSCDBGasGainCorrection_h

#include <iosfwd>
#include <vector>

class CSCDBGasGainCorrection{
 public:
  CSCDBGasGainCorrection(){}
  ~CSCDBGasGainCorrection(){}

  struct Item{
    float gainCorr;
  };

  typedef std::vector<Item> GasGainContainer;
  GasGainContainer gasGainCorr;

  const Item & item( int index ) const { return gasGainCorr[index]; }
  float value( int index ) const { return gasGainCorr[index].gainCorr; }
};

std::ostream & operator<<(std::ostream & os, const CSCDBGasGainCorrection & cscdb);

#endif
