#ifndef CSCRecHitD_CSCHitFromStripOnly_h
#define CSCRecHitD_CSCHitFromStripOnly_h

/** \class CSCHitFromStripOnly (old comment below)
 *
 * Search for strip with ADC output exceeding theThresholdForAPeak.  For each of these strips,
 * build a cluster of strip of size theClusterSize (typically 5 strips).  Finally, make
 * a Strip Hit out of these clusters by finding the center-of-mass position of the hit
 * The DetId, strip hit position, and peaking time are stored in a CSCStripHit collection.
 *
 * Be careful with the ME_1/a chambers: the 48 strips are ganged into 16 channels,
 * Each channel contains signals from three strips, each separated by 16 strips from the next.
 *
 */
#include "DataFormats/MuonDetId/interface/CSCDetId.h"

#include "RecoLocalMuon/CSCRecHitD/src/CSCStripData.h"
#include "RecoLocalMuon/CSCRecHitD/src/CSCStripHitData.h"
#include "RecoLocalMuon/CSCRecHitD/src/CSCStripHit.h"
#include "RecoLocalMuon/CSCRecHitD/src/CSCRecoConditions.h"

#include "DataFormats/CSCDigi/interface/CSCStripDigiCollection.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <vector>

class CSCLayer;
class CSCStripDigi;
class CSCPedestalChoice;

class CSCHitFromStripOnly 
{
  
 public:

  typedef std::vector<CSCStripData> PulseHeightMap;
  
  explicit CSCHitFromStripOnly( const edm::ParameterSet& ps );
  
  ~CSCHitFromStripOnly();
  
  std::vector<CSCStripHit> runStrip( const CSCDetId& id, const CSCLayer* layer, const CSCStripDigiCollection::Range& rstripd );

  void setConditions( const CSCRecoConditions* reco ) {
    recoConditions_ = reco;
  } 
 
  bool ganged() { return ganged_;}
  void setGanged( bool ig ) { ganged_ = ig;}

 private:
	
  /// Store SCA pulseheight information from strips in digis of one layer
  void fillPulseHeights( const CSCStripDigiCollection::Range& rstripd );  

  /// Find local maxima
  void findMaxima(const CSCDetId& id);  
  // What we call a peak
  bool isPeakOK(int iStrip, float heightCluster);  

  /// Make clusters using local maxima
  float makeCluster( int centerStrip );

  /// 
  CSCStripHitData makeStripData( int centerStrip, int offset );

  /// Is either neighbour 'bad'?
  bool isNearDeadStrip(const CSCDetId& id, int centralStrip); 

  /// Is the strip 'bad'?
  bool isDeadStrip(const CSCDetId& id, int centralStrip); 

  /// Find position of hit in strip cluster in terms of strip #
  float findHitOnStripPosition( const std::vector<CSCStripHitData>& data, const int& centerStrip );
  

// MEMBER DATA

  // Hold pointers to current layer, conditions data
  CSCDetId id_;    
  const CSCLayer * layer_;
  const CSCRecoConditions* recoConditions_;
  // Number of strips in layer
  unsigned nstrips_;
  // gain correction weights and crosstalks read in from conditions database.
  float gainWeight[80];

  // The specific pedestal calculator
	CSCPedestalChoice* calcped_;

  // The cuts for forming the strip hits are described in the config file
  bool useCalib;
  static const int theClusterSize = 3;
  float theThresholdForAPeak;
  float theThresholdForCluster;


  // working buffer for sca pulseheights
  PulseHeightMap thePulseHeightMap;

  std::vector<int> theMaxima;
  std::vector<int> theConsecutiveStrips;//... with charge for a given maximum
  std::vector<int> theClosestMaximum; // this is number of strips to the closest other maximum

  // Variables entering the CSCStripHit construction:
  int tmax_cluster; // Peaking time for strip hit, in time bin units
  int clusterSize;
  std::vector<float> strips_adc;
  std::vector<float> strips_adcRaw;
  std::vector<int> theStrips;

  bool ganged_; // only True if ME1/1A AND it is ganged
  
};

#endif

