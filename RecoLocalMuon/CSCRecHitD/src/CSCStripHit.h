#ifndef CSCRecHitD_CSCStripHit_H
#define CSCRecHitD_CSCStripHit_H

/**
 * \class CSCStripHit
 *
 * A (1-dim) hit reconstructed from strip data alone
 *
 */

#include "DataFormats/MuonDetId/interface/CSCDetId.h"

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
               const StripHitADCContainer& s_adcRaw,
               const int& numberOfConsecutiveStrips,
               const int& closestMaximum,
               const bool& isNearDeadStrip);

  ~CSCStripHit();

  /// Required for storage in RangeMap
  CSCStripHit* clone() const { return new CSCStripHit( *this ); }

  /// Strip Hit is in this DetId
  CSCDetId cscDetId() const { return theDetId; }

  /// Strip hit position (centroid)
  float sHitPos() const { return theStripHitPosition; }

  /// Strip hit maximum time bin
  int tmax() const { return theStripHitTmax; }

  /// The strips used in cluster to produce strip hit
  const ChannelContainer& strips() const { return theStrips; }

  /// the ADC counts for each of the strip within cluster
  const StripHitADCContainer& s_adc() const { return theStripHitADCs; }

  /// the raw ADC counts for each of the strip within cluster
  const StripHitADCContainer& s_adcRaw() const { return theStripHitRawADCs; }

  /// Number of consecutive strips with charge
  int numberOfConsecutiveStrips() const { return theConsecutiveStrips; }

  /// Number of strips to the closest other maximum
  int closestMaximum() const { return theClosestMaximum; }

  /// is a neighbouring string a dead strip?
  bool isNearDeadStrip() const {return isDeadStripAround; };

  
private:
  CSCDetId theDetId;
  float theStripHitPosition;
  int theStripHitTmax;
  ChannelContainer theStrips;
  StripHitADCContainer theStripHitADCs;  
  StripHitADCContainer theStripHitRawADCs;  
  int theConsecutiveStrips;
  int theClosestMaximum;
  bool isDeadStripAround;

};



#endif

