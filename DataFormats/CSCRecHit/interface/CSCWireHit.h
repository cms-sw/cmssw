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
  CSCWireHit( const CSCDetId& id, const int& wire_group );

  ~CSCWireHit();

  /// CSCWireHit base class interface
  CSCWireHit* clone() const { return new CSCWireHit( *this ); }

//  /// TrackingRecHit base class interface  --> do I need this ???
//  DetId geographicalId() const { return theDetId; }
    CSCDetId cscDetId() const { return theDetId; }

  /// Container 1-D wire hit position expressed in terms of wiregroup #
  int wire_group() const { return theWireHitPosition; }

private:
  CSCDetId theDetId;
  int theWireHitPosition;
};


#endif

