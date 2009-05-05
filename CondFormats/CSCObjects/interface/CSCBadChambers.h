#ifndef CSCBadChambers_h
#define CSCBadChambers_h

#include "DataFormats/MuonDetId/interface/CSCDetId.h"

#include <vector>

class CSCBadChambers{
 public:
  CSCBadChambers() : numberOfBadChambers_( 0 ), skipBadChambers_( true ), chambers_( std::vector<int>() ){};
  ~CSCBadChambers(){};

  /// How many bad chambers are there>
  int numberOfChambers() const { return numberOfBadChambers_; }

  /// Return the flag for skipping bad chambers or not
  bool skipBadChambers() const { return skipBadChambers_; }

  /// Return the container of bad chambers
  std::vector<int> chambers() const { return chambers_; }

  /// Set the flag for skipping bad chambers or not (activate/deactivate call to isInBadChamber)
  void switchSkipBadChambers( bool onoff ) { skipBadChambers_ = onoff; }

  /// Is the gven chamber flagged as bad?
  bool isInBadChamber( const CSCDetId& id ) const;

 private:
  int numberOfBadChambers_;
  bool skipBadChambers_;
  std::vector<int> chambers_;
};

#endif
