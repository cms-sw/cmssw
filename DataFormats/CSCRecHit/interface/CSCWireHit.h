#ifndef DataFormats_CSCWireHit_H
#define DataFormats_CSCWireHit_H

/**
 * \class CSCSWireHit
 *
 * Yields the position in terms wire group # of a 1-D reconstructed 
 * wire hit in one layer of an Endcap Muon CSC.
 * 
 *
 * \author Dominique Fortin - UCR
 *
 */

#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <vector>
#include <iosfwd>

class CSCWireHit 
{

public:

  CSCWireHit();
  CSCWireHit( const CSCDetId& id, const float& wgroup, const int& tmax );

  ~CSCWireHit();

  /// CSCWireHit base class interface
  CSCWireHit* clone() const { return new CSCWireHit( *this ); }

  /// Position of the wire hit in CSC
  CSCDetId cscDetId() const { return theDetId; }

  /// The wire hit position expressed in terms of wiregroup #
  float wgrouppos() const { return theHitWirePosition; }

  /// The time for the wire hit
  int tmax() const { return theHitTmax; }
 
private:
  CSCDetId theDetId;
  float theHitWirePosition;
  int theHitTmax;
};


// Output operator for CSCWireHit
//std::ostream& operator<<(std::ostream& os, const CSCWireHit& rh);


#endif

