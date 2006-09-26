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

  typedef std::vector<float> StripHitADCContainer;


  CSCStripHit();
  CSCStripHit( const CSCDetId& id, const float& sHitPos, const int& tmax, 
               const int& clusterSize, const StripHitADCContainer& s_adc );

  ~CSCStripHit();

  /// CSCStripHit base class interface
  CSCStripHit* clone() const { return new CSCStripHit( *this ); }

  /// Strip Hit posion in terms of DetId
    CSCDetId cscDetId() const { return theDetId; }

  /// Strip hit position expressed in terms of 1/2 strip #
  float sHitPos() const { return theStripHitPosition; }

  /// Strip hit maximum time bin
  int tmax() const { return theStripHitTmax; }

  /// Number of strips in cluster to produce strip hit
  int clusterSize() const { return theStripHitClusterSize; }


  /// the ADC counts for each of the strip within cluster
  const StripHitADCContainer& s_adc() const { return theStripHitADCs; }


  
private:
  CSCDetId theDetId;
  float theStripHitPosition;
  int theStripHitTmax;
  int theStripHitClusterSize;
  StripHitADCContainer theStripHitADCs;  

};



#endif

