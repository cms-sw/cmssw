#ifndef CSCRecHitD_CSCHitFromStripOnly_h
#define CSCRecHitD_CSCHitFromStripOnly_h

/** \class CSCHitFromStripOnly
 *
 * Search for strip with ADC output exceeding theThresholdForAPeak.  For each of these strips,
 * build a cluster of strip of size theClusterSize (typically 5 strips).  Finally, make
 * a Strip Hit out of these clusters by finding the center-of-mass position of the hit
 * The DetId, strip hit position, and peaking time are stored in a CSCStripHit collection.
 *
 * Here one has to be careful with the ME_1/a chambers:  in MC, digis are produced only for the first 16
 * strips, so one has to account for the ganging in groups of 3.
 *
 * In data, the ME_11 digis are stored in the same collection, so one has to untangle the output from
 * the ME_1a and ME_1b strips.  64 readouts from ME_1b, 16 from ME_1a.  Will have to figure out if ME_1a comes
 * first, and then the 64 ME_1b...
 *
 * \author Dominique Fortin - UCR
 *
 */
//---- Possible changes from Stoyan Stoynev - NU
#include <RecoLocalMuon/CSCRecHitD/src/CSCStripData.h>
#include <RecoLocalMuon/CSCRecHitD/src/CSCStripHitData.h>
#include <RecoLocalMuon/CSCRecHitD/src/CSCStripHit.h>

#include <DataFormats/CSCDigi/interface/CSCStripDigiCollection.h>

#include <FWCore/Framework/interface/Frameworkfwd.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>

#include <vector>

class CSCLayer;
class CSCChamberSpecs;
class CSCLayerGeometry;
class CSCDBGains;
class CSCStripDigi;
class CSCPeakBinOfStripPulse;
class CSCStripGain;


class CSCHitFromStripOnly 
{
  
 public:

  typedef std::vector<CSCStripData> PulseHeightMap;
  
  explicit CSCHitFromStripOnly( const edm::ParameterSet& ps );
  
  ~CSCHitFromStripOnly();
  
  std::vector<CSCStripHit> runStrip( const CSCDetId& id, const CSCLayer* layer, const CSCStripDigiCollection::Range& rstripd );

  void setCalibration( float gainAvg, const CSCDBGains* gains ) { 
    globalGainAvg  = gainAvg;
    gains_ = gains; 
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
  int ClusterSize;
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


  /* Cache calibrations for current event
   *
   */
  const CSCDBGains*       gains_;
  float globalGainAvg;

  CSCPeakBinOfStripPulse* pulseheightOnStripFinder_;
  CSCStripGain*           stripGain_;

};

#endif

