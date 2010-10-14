#ifndef CSCRecHitD_CSCWireHit_H
#define CSCRecHitD_CSCWireHit_H

/**
 * \class CSCSWireHit
 *
 * Yields the position in terms wire group # of a 1-D reconstructed 
 * wire hit in one layer of an Endcap Muon CSC.
 *
 */

#include <DataFormats/MuonDetId/interface/CSCDetId.h>

#include <vector>
#include <iosfwd>

class CSCWireHit 
{

public:

  typedef std::vector<int> ChannelContainer;

  CSCWireHit();
  CSCWireHit( const CSCDetId& id, const float& wHitPos, ChannelContainer& wgroups, const int& tmax,
  const bool& isNearDeadWG );

  ~CSCWireHit();

  /// CSCWireHit base class interface
  CSCWireHit* clone() const { return new CSCWireHit( *this ); }

  /// Position of the wire hit in CSC
  CSCDetId cscDetId() const { return theDetId; }

  /// The wire hit position expressed in terms of wire #
  float wHitPos() const { return theWireHitPosition; }

  /// The wire groups used for forming the cluster
  //ChannelContainer wgroups() const { return theWgroups; }
  ChannelContainer wgroups() const { return theWgroupsLowBits; }

  /// The BX number
  ChannelContainer wgroupsBX() const { return theWgroupsHighBits; }

  /// The BX + wire group number
  ChannelContainer wgroupsBXandWire() const { return theWgroups; }

  /// The timing for the wire hit
  int tmax() const { return theWireHitTmax; }

  /// is a neighbouring WG a dead WG?
  bool isNearDeadWG() const {return isDeadWGAround; };

  /// Print content of the wirehit
  void print() const;

private:
  CSCDetId theDetId;
  float theWireHitPosition;
  ChannelContainer theWgroups; /// BX and wire group number combined
  ChannelContainer theWgroupsHighBits; /// to extract BX
  ChannelContainer theWgroupsLowBits; /// to extract the wire group number
  int theWireHitTmax;
  bool isDeadWGAround;
};



#endif

