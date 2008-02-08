#ifndef CSCRecHitD_CSCRecHitDBuilder_h 
#define CSCRecHitD_CSCRecHitDBuilder_h 


/** \class CSCRecHitDBuilder 
 *
 * Algorithm to build 2-D RecHit from wire and strip digis
 * in endcap muon CSCs by implementing a 'build' function 
 * required by CSCRecHitDProducer.
 *
 * The builder goes through many stages before building 2-D hits:
 * 1) It finds wire clusters and form wire hits which it stores in CSCWireHit.
 * 2) From these wire hits, it builds pseudo-wire segments to clean up
 *    the wire hit collection from noisy hits.  Only the hits falling on
 *    the segment or far away from existing segments are retained.
 * 1) It then finds strip cluster and hits which it stores in CSCStripHit.
 * 2) Similary to the wire hits, segments are build using the strip hits.
 *    Because of the trapezoidal geometry of the strips, all strip hits falling
 *    close to the pseudo-strip segments are retained.
 *
 * \author Stoyan Stoynev - NU
 *
 */

#include <DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h>
#include <DataFormats/CSCDigi/interface/CSCStripDigiCollection.h>
#include <DataFormats/CSCDigi/interface/CSCWireDigiCollection.h>

#include <FWCore/ParameterSet/interface/ParameterSet.h>

class CSCLayer;
class CSCGeometry;
class CSCDBGains;
class CSCDBCrosstalk;
class CSCDBNoiseMatrix;
class CSCDetId;
class CSCHitFromStripOnly;
class CSCHitFromWireOnly;
class CSCWireSegments;
class CSCStripSegments;
class CSCMake2DRecHit;

class CSCRecHitDBuilder
{
 public:
  
  /** Configure the algorithm via ctor.
   * Receives ParameterSet percolated down from EDProducer
   * which owns this Builder.
   */
  explicit CSCRecHitDBuilder( const edm::ParameterSet& ps);
  
  ~CSCRecHitDBuilder();
  
  /** Find digis in each CSCLayer, build strip and wire proto-hits in 
   *  each layer from which pseudo-segments are build to select hits.
   *  Then, strip/wire hits are combined to form 2-D hits, whereas
   *  remaining "good" strip and wire only hits are also stored  
   *  into output collection.
   *
   */
  
  void build( const CSCStripDigiCollection* stripds, const CSCWireDigiCollection* wireds,
	      CSCRecHit2DCollection& oc );
  
  /*
   * Cache pointer to geometry and calibration constants so they can be
   * redistributed further downstream
   */
  void setGeometry   ( const CSCGeometry* geom ) {geom_ = geom;}
  void setCalibration( float gainAvg,
	               const CSCDBGains* gains,
                       const CSCDBCrosstalk* xtalk,
                       const CSCDBNoiseMatrix* noise ) {
    gAvg_  = gainAvg;
    gains_ = gains;
    xtalk_ = xtalk;
    noise_ = noise;
  }

  const CSCLayer* getLayer( const CSCDetId& detId );


 private:

  bool isData;
  bool useCalib;
  bool debug;  
  int stripWireDeltaT;
  bool useCleanStripCollection;
  bool useCleanWireCollection;
  bool makePseudo2DHits;

  /**
   *  The Program first constructs proto wire/strip hits which
   *  it stores in a special collection.  Proto strip/wire segments
   *  are then build from these hits and allow to clean up up the list of hits.
   */
  CSCHitFromStripOnly*   HitsFromStripOnly_;
  CSCHitFromWireOnly*    HitsFromWireOnly_;
  //CSCWireSegments*       HitsFromWireSegments_;  
  //CSCStripSegments*      HitsFromStripSegments_;  
  CSCMake2DRecHit*       Make2DHits_;

  /*
   * Cache geometry and calibrations for current event
   */
  const CSCGeometry* geom_;
  float gAvg_;
  const CSCDBGains* gains_;
  const CSCDBCrosstalk* xtalk_;
  const CSCDBNoiseMatrix* noise_;

};

#endif
