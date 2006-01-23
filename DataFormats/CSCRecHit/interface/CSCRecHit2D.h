#ifndef CSCRecHit2D_H
#define CSCRecHit2D_H

/**
 * \class CSCRecHit2D
 * Describes a 2-dim reconstructed hit in one layer of an Endcap Muon CSC.
 *
 * \author Tim Cox
 *
 */

#include <DataFormats/TrackingRecHit/interface/RecHit2DLocalPos.h>
#include <vector>

class CSCRecHit2D : public RecHit2DLocalPos
{
public:

  typedef std::vector<int> ChannelContainer;

  CSCRecHit2D();
  CSCRecHit2D( const DetId& id, const GeomDet* det, 
               const LocalPoint& pos, const LocalError& err, 
	       const ChannelContainer& channels,
               float chi2, float prob );
  ~CSCRecHit2D();

  /// RecHit2DLocalPos base class interface
  CSCRecHit2D* clone() const { return new CSCRecHit2D( * this); }
  LocalPoint localPosition() const { return theLocalPosition;}
  LocalError localPositionError() const { return theLocalError;}

  /// TrackingRecHit base class interface
  const GeomDet& det() const { return *theDet; }
  DetId geographicalId() const { return theDetId; }

  /// Probability from fit during rechit build
  float prob() const { return theProb; }

  /// Chi-squared from fit during rechit build
  float chi2() const { return theChi2; }


  /// Container of strip channel numbers comprising the rechit
  const ChannelContainer& channels() const {
    return theChaCo;
  }

  // No longer handles global values?
  // GlobalPoint globalPosition() const;

  //  Useful when building segments...
  //  bool nearby(const CSCRecHit2D& other, float maxDeltaRPhi);
  //  bool nearby(float otherX, float maxDeltaRPhi);

private:
  DetId theDetId;
  const GeomDet* theDet;
  LocalPoint theLocalPosition;
  LocalError theLocalError;
  ChannelContainer theChaCo;
  float theChi2;
  float theProb;

};

#endif

