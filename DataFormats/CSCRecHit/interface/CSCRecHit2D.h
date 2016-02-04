#ifndef DataFormats_CSCRecHit2D_H
#define DataFormats_CSCRecHit2D_H

/**
 * \class CSCRecHit2D
 * Describes a 2-dim reconstructed hit in one layer of an Endcap Muon CSC.
 *
 * \author Tim Cox et al.
 *
 */
#include "DataFormats/Common/interface/RangeMap.h"
#include <DataFormats/TrackingRecHit/interface/RecHit2DLocalPos.h>
#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <vector>
#include <map>
#include <iosfwd>

class CSCRecHit2D : public RecHit2DLocalPos {

public:

  typedef std::vector<int> ChannelContainer;
  typedef edm::RangeMap<int, std::vector<float> > ADCContainer;
  
  enum SharedInputType {all = TrackingRecHit::all, some = TrackingRecHit::some, allWires, someWires, allStrips, someStrips};

  CSCRecHit2D();

  CSCRecHit2D( const CSCDetId& id, 
               const LocalPoint& pos, const LocalError& err, 
	       const ChannelContainer& channels,
	       const ADCContainer& adcs,
	       const ChannelContainer& wgroups,
               float tpeak,
               float posInStrip,
               float errInStrip,
	       int quality,
               short int badStrip=0, short int badWireGroup=0,
               int scaledWireTime=0 ); 
	
  ~CSCRecHit2D();

  /// RecHit2DLocalPos base class interface
  CSCRecHit2D* clone() const { return new CSCRecHit2D( *this ); }
  LocalPoint localPosition() const { return theLocalPosition; }
  LocalError localPositionError() const { return theLocalError; }
  CSCDetId cscDetId() const { return geographicalId(); }

  /// Extracting strip channel numbers comprising the rechit
   const ChannelContainer& channels() const { return theStripsLowBits; } /// L1A
  /// const ChannelContainer& channels() const { return theStrips; }
  
  /// Extract the L1A phase bits from the StripChannelContainer
   const ChannelContainer& channelsl1a() const { return theStripsHighBits; } /// L1A
  
  /// Container of the L1A+Channels comprising the rechit
   const ChannelContainer& channelsTotal() const { return theStrips; } /// L1A

  /// Map of strip ADCs for strips comprising the rechit
  const ADCContainer& adcs() const { return theADCs; }

  /// Container of wire groups comprising the rechit
  //const ChannelContainer& wgroups() const { return theWireGroups; }
  const ChannelContainer& wgroups() const { return theWgroupsLowBits; }

  /// The BX number
  ChannelContainer wgroupsBX() const { return theWgroupsHighBits; }

  /// The BX + wire group number
  ChannelContainer wgroupsBXandWire() const { return theWireGroups; }

  /// Fitted peaking time
  float tpeak() const { return theTpeak; }

  /// The estimated position within the strip
  float positionWithinStrip() const { return thePositionWithinStrip; };

  /// The uncertainty of the estimated position within the strip
  float errorWithinStrip() const { return theErrorWithinStrip;} ;

  /// quality flag of the reconstruction
  int quality() const { return theQuality;}

  /// flags for involvement of 'bad' channels
  short int badStrip() const { return theBadStrip; }
  short int badWireGroup() const { return theBadWireGroup; }
  
  // Calculated wire time in ns
  float wireTime() const { return (float)theScaledWireTime/100.; }

  /// Returns true if the two TrackingRecHits are using the same input information, false otherwise.  In this case, looks at the geographical ID and channel numbers for strips and wires.
  virtual bool sharesInput(const TrackingRecHit *other, TrackingRecHit::SharedInputType what) const;
  
  /// Returns true if the two TrackingRecHits are using the same input information, false otherwise.  In this case, looks at the geographical ID and channel numbers for strips and wires.
  bool sharesInput(const TrackingRecHit *other, CSCRecHit2D::SharedInputType what) const;

  /// Returns true if the two CSCRecHits are using the same input information, false otherwise.  In this case, looks at the geographical ID and channel numbers for strips and wires.
  bool sharesInput(const  CSCRecHit2D *otherRecHit, CSCRecHit2D::SharedInputType what) const;
   
   /// Print the content of the RecHit2D including L1A (for debugging)
   void print() const;	

private:
	
  LocalPoint theLocalPosition;
  LocalError theLocalError;
  ChannelContainer theStrips;
  ADCContainer theADCs;
  ChannelContainer theWireGroups;
  float theTpeak;
  float thePositionWithinStrip; 
  float theErrorWithinStrip;
  int theQuality;
  short int theBadStrip;
  short int theBadWireGroup;
  int theScaledWireTime;
  ChannelContainer theStripsLowBits; /// L1A
  ChannelContainer theStripsHighBits; /// L1A
  ChannelContainer theWgroupsHighBits; /// BX
  ChannelContainer theWgroupsLowBits; ///  BX
 
};

/// Output operator for CSCRecHit2D
std::ostream& operator<<(std::ostream& os, const CSCRecHit2D& rh);

#endif

