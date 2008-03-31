#ifndef CSCRecHitD_CSCStripHit_H
#define CSCRecHitD_CSCStripHit_H

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

  typedef std::vector<float> StripHitADCContainer;
  typedef std::vector<int> ChannelContainer;

  CSCStripHit();
  CSCStripHit( const CSCDetId& id, 
               const float& sHitPos, 
               const int& tmax, 
               const ChannelContainer& strips, 
               const StripHitADCContainer& s_adc,
               const int& numberOfConsecutiveStrips,
               const int& closestMaximum);

  ~CSCStripHit();

  /// CSCStripHit base class interface
  CSCStripHit* clone() const { return new CSCStripHit( *this ); }

  /// Strip Hit posion in terms of DetId
  CSCDetId cscDetId() const { return theDetId; }

  /// Strip hit position expressed in terms of 1/2 strip #
  float sHitPos() const { return theStripHitPosition; }

  /// Strip hit maximum time bin
  int tmax() const { return theStripHitTmax; }

  /// The strips used in cluster to produce strip hit
  const ChannelContainer& strips() const { return theStrips; }

  /// the ADC counts for each of the strip within cluster
  const StripHitADCContainer& s_adc() const { return theStripHitADCs; }

  /// Number of consecutive strips with charge
  int numberOfConsecutiveStrips() const { return theConsecutiveStrips; }

  /// Number of strips to the closest other maximum
  int closestMaximum() const { return theClosestMaximum; }

  
private:
  CSCDetId theDetId;
  float theStripHitPosition;
  int theStripHitTmax;
  ChannelContainer theStrips;
  StripHitADCContainer theStripHitADCs;  
  int theConsecutiveStrips;
  int theClosestMaximum;

};



#endif

