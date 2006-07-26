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
  CSCStripHit( const CSCDetId& id, const float& strip_pos );

  ~CSCStripHit();

  /// CSCStripHit base class interface
  CSCStripHit* clone() const { return new CSCStripHit( *this ); }

//  /// TrackingRecHit base class interface  --> do I need this ???
//  DetId geographicalId() const { return theDetId; }
    CSCDetId cscDetId() const { return theDetId; }

  /// Container 1-D strip hit position expressed in terms of strip #
  float strip_pos() const { return theStripHitPosition; }

private:
  CSCDetId theDetId;
  float theStripHitPosition;
};


#endif

