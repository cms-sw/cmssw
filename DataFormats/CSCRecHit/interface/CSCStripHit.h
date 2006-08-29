#ifndef DataFormats_CSCStripHit_H
#define DataFormats_CSCStripHit_H

/**
 * \class CSCStripHit
 *
 * Yields the position in terms of 1/2 strip # of a 1-D reconstructed 
 * strip hit in one layer of an Endcap Muon CSC.
 * 
 *
 * \author Dominique Fortin - UCR
 *
 */

#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <vector>
#include <iosfwd>

class CSCStripHit 
{

public:

  CSCStripHit();
  CSCStripHit( const CSCDetId& id, const float& halfStripPos, const int& tmax );

  ~CSCStripHit();

  /// CSCStripHit base class interface
  CSCStripHit* clone() const { return new CSCStripHit( *this ); }

  /// Strip Hit posion in terms of DetId
    CSCDetId cscDetId() const { return theDetId; }

  /// Strip hit position expressed in terms of 1/2 strip #
  float halfStripPos() const { return theHitHalfStripPosition; }

  /// Strip hit timing
  int tmax() const { return theHitTmax; }
  
private:
  CSCDetId theDetId;
  float theHitHalfStripPosition;
  int theHitTmax;
};

// Output operator for CSCStripHit
//std::ostream& operator<<(std::ostream& os, const CSCStripHit& rh);


#endif

