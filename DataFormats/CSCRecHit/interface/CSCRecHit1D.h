#ifndef DataFormats_CSCRecHit1D_H
#define DataFormats_CSCRecHit1D_H

/**
 * \class CSCRecHit1D
 *
 * Describes a 1-D reconstructed hit in one layer of an Endcap Muon CSC.
 *
 * \author Dominique Fortin - UCR
 *
 */

#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <vector>
#include <iosfwd>

class CSCRecHit1D 
{

public:

//  typedef std::vector<float> ChannelContainer;  // Don't need this ...

  CSCRecHit1D();
  CSCRecHit1D( const CSCDetId& id, const float& channel );

  ~CSCRecHit1D();

  /// RecHit1DLocalPos base class interface
  CSCRecHit1D* clone() const { return new CSCRecHit1D( *this ); }

  /// TrackingRecHit base class interface
  DetId geographicalId() const { return theDetId; }
  CSCDetId cscDetId() const { return theDetId; }

  /// Container of channel number where the 1-D rechit (expressed in terms of strip or wiregroup #)
  float channel() const { return theChannel; }

private:
  CSCDetId theDetId;
  float theChannel;
};


#endif

