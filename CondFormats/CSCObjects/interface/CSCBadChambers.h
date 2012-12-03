#ifndef CSCBadChambers_h
#define CSCBadChambers_h

#include <vector>

class CSCBadChambers{
 public:
  CSCBadChambers() : numberOfBadChambers( 0 ), chambers( std::vector<int>() ) {};
  CSCBadChambers(int nch, std::vector<int> ch ) : numberOfBadChambers( nch ), chambers( ch ) {};
  ~CSCBadChambers(){};

  /// How many bad chambers are there>
  int numberOfChambers() const { return numberOfBadChambers; }

  /// Return the container of bad chambers
  std::vector<int> container() const { return chambers; }

  /// Is the chamber  with index 'ichamber' flagged as bad?
  bool isInBadChamber( int ichamber ) const;

 private:
  int numberOfBadChambers;
  std::vector<int> chambers;
};

#endif
