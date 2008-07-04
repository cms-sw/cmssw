#ifndef CSCRecHitD_CSCHitFromStripOnly_h
#define CSCRecHitD_CSCHitFromStripOnly_h

/** \class CSCHitFromStripOnly
 *
 * Search for strip with ADC output exceeding theThresholdForAPeak.  For each of these strips,
 * build a cluster of strip of size theClusterSize (typically 5 strips).  Finally, make
 * a Strip Hit out of these clusters by finding the center-of-mass position of the hit
 * The DetId, strip hit position, and peaking time are stored in a CSCStripHit collection.
 *
 * Be careful with the ME_1/a chambers: the 48 strips are ganged into 16 channels,
 * Each channel contains signals from three strips, each separated by 16 strips from the next.
 * This is the same for real data and for the MC. (This ME1a info is from Tim Cox.)
 *
 * \author Dominique Fortin - UCR
 *
 */
//---- Possible changes from Stoyan Stoynev - NU
#include <RecoLocalMuon/CSCRecHitD/src/CSCStripData.h>
#include <RecoLocalMuon/CSCRecHitD/src/CSCStripHitData.h>
#include <RecoLocalMuon/CSCRecHitD/src/CSCStripHit.h>
#include <RecoLocalMuon/CSCRecHitD/src/CSCRecoConditions.h>

#include <DataFormats/CSCDigi/interface/CSCStripDigiCollection.h>

#include <FWCore/Framework/interface/Frameworkfwd.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>

#include <vector>

class CSCLayer;
class CSCChamberSpecs;
class CSCLayerGeometry;
class CSCStripDigi;
class CSCPeakBinOfStripPulse;


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

 protected:
  
  /// Go through strip in layers and build a table with 
  void fillPulseHeights( const CSCStripDigiCollection::Range& rstripd );  

  /// Find local maxima
  void findMaxima();    

  /// Make clusters using local maxima
  float makeCluster( int centerStrip );

  std::vector<int> theMaxima;

  PulseHeightMap thePulseHeightMap;
  
  /// Find position of hit in strip cluster in terms of strip #
  float findHitOnStripPosition( const std::vector<CSCStripHitData>& data, const int& centerStrip );
  
  CSCDetId id_;    
  const CSCLayer * layer_;
  const CSCLayerGeometry * layergeom_;
  const CSCChamberSpecs * specs_;
  
 private:
  
  CSCStripHitData makeStripData( int centerStrip, int offset );


  // Variables entering the CSCStripHit construction:
  int tmax_cluster;
  int clusterSize;
  std::vector<float> strips_adc;  
  std::vector<int> theStrips;
  
  // The cuts for forming the strip hits are described in the data/.cfi file
  bool debug;
  bool useCalib;
  int theClusterSize;
  float theThresholdForAPeak;
  float theThresholdForCluster;
  bool useCleanStripCollection;
  //bool isData;

  /// These are the gain correction weights and X-talks read in from database.
  float gainWeight[80];

  // Peaking time for strip hit
  int TmaxOfCluster;            // in time bins;
  // Number of strips in layer
  unsigned Nstrips;

  /// Hold pointer to current conditions data
  const CSCRecoConditions* recoConditions_;

  CSCPeakBinOfStripPulse* pulseheightOnStripFinder_;
};

#endif

