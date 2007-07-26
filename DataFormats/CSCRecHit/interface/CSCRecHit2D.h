#ifndef DataFormats_CSCRecHit2D_H
#define DataFormats_CSCRecHit2D_H

/**
 * \class CSCRecHit2D
 * Describes a 2-dim reconstructed hit in one layer of an Endcap Muon CSC.
 *
 * \author Tim Cox
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

  CSCRecHit2D();
  //  CSCRecHit2D( const DetId& id, const GeomDet* det, 
  CSCRecHit2D( const CSCDetId& id, 
               const LocalPoint& pos, const LocalError& err, 
	       const ChannelContainer& channels,
	       const ADCContainer& adcs,
	       const ChannelContainer& wgroups,
               const float tpeak, 
               float chi2, 
               float prob );
  ~CSCRecHit2D();

  /// RecHit2DLocalPos base class interface
  CSCRecHit2D* clone() const { return new CSCRecHit2D( *this ); }
  LocalPoint localPosition() const { return theLocalPosition; }
  LocalError localPositionError() const { return theLocalError; }

  CSCDetId cscDetId() const { return geographicalId(); }

  /// Probability from fit during rechit build
  float prob() const { return theProb; }

  /// Chi-squared from fit during rechit build
  float chi2() const { return theChi2; }

  /// Fitted peaking time
  float tpeak() const { return theTpeak; }

  /// Container of strip channel numbers comprising the rechit
  const ChannelContainer& channels() const {
    return theChaCo;
  }

  /// Map of strip ADCs for strips comprising the rechit
  const ADCContainer& adcs() const {
    return theADCs;
  }

  /// Container of wire groups comprising the rechit
  const ChannelContainer& wgroups() const {
    return theWireGroups;
  }


  // To handle global values must use DetId to identify Det, hence Surface, which can transform from local
  // GlobalPoint globalPosition() const;

  //  Useful when building segments...
  //  bool nearby(const CSCRecHit2D& other, float maxDeltaRPhi);
  //  bool nearby(float otherX, float maxDeltaRPhi);

private:

  //  const GeomDet* theDet;
  LocalPoint theLocalPosition;
  LocalError theLocalError;
  ChannelContainer theChaCo;
  ADCContainer theADCs;
  ChannelContainer theWireGroups;
  float theTpeak;
  float theChi2;
  float theProb;
};

/// Output operator for CSCRecHit2D
std::ostream& operator<<(std::ostream& os, const CSCRecHit2D& rh);

#endif

