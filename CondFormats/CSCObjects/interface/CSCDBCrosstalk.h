#ifndef CSCObjects_CSCDBCrosstalk_h
#define CSCObjects_CSCDBCrosstalk_h

#include <iosfwd>
#include <vector>

class CSCDBCrosstalk
{
 public:
  CSCDBCrosstalk() {}
  ~CSCDBCrosstalk() {}

  struct Item{
    short int xtalk_slope_right;
    short int xtalk_intercept_right;
    short int xtalk_slope_left;
    short int xtalk_intercept_left;
  };
  int factor_slope;
  int factor_intercept;

  enum factors{FSLOPE=10000000, FINTERCEPT=100000};

  typedef std::vector<Item> CrosstalkContainer;
  CrosstalkContainer crosstalk;

  const Item & item( int index ) const { return crosstalk[index]; }
  short int rslope( int index ) const { return crosstalk[index].xtalk_slope_right; }
  short int rinter( int index ) const { return crosstalk[index].xtalk_intercept_right; }
  short int lslope( int index ) const { return crosstalk[index].xtalk_slope_left; }
  short int linter( int index ) const { return crosstalk[index].xtalk_intercept_left; }
  int sscale() const { return factor_slope; }
  int iscale() const { return factor_intercept; }
};

std::ostream & operator<<(std::ostream & os, const CSCDBCrosstalk & cscdb);

#endif

