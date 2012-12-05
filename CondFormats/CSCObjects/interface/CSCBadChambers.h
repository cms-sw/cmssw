#ifndef CSCBadChambers_h
#define CSCBadChambers_h

#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <vector>

class CSCBadChambers{
 public:
  typedef uint16_t IndexType;

  CSCBadChambers() : numberOfBadChambers( 0 ), chambers( std::vector<int>() ) {};
  CSCBadChambers(int nch, std::vector<int> ch ) : numberOfBadChambers( nch ), chambers( ch ) {};
  ~CSCBadChambers(){};

  /// How many bad chambers are there>
  int numberOfChambers() const { return numberOfBadChambers; }

  /// Return the container of bad chambers
  std::vector<int> container() const { return chambers; }

  /// Is the chamber  with index 'ichamber' flagged as bad?
  bool isInBadChamber( IndexType ichamber ) const;

  /// Is the chamber  with CSCDetId 'id' flagged as bad?
  bool isInBadChamber( const CSCDetId& id ) const;

  IndexType startChamberIndexInEndcap(IndexType ie, IndexType is, IndexType ir) const
  {
    const IndexType nschin[32] =
      { 1,37,73,1,        109,127,0,0,  163,181,0,0,  217,469,0,0,
        235,271,307,235,  343,361,0,0,  397,415,0,0,  451,505,0,0 };
    return nschin[(ie - 1)*16 + (is - 1)*4 + ir - 1];
  }

  IndexType chamberIndex(IndexType ie, IndexType is, IndexType ir, IndexType ic) const
  {
    return startChamberIndexInEndcap(ie, is, ir) + ic - 1; // -1 so start index _is_ ic=1
  }

 private:
  int numberOfBadChambers;
  std::vector<int> chambers;
};

#endif
