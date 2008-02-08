#ifndef CSCRecHitD_CSCMake2DRecHit_h
#define CSCRecHitD_CSCMake2DRecHit_h
 
/** \class CSCMake2DRecHit
 *
 * The overlap between strip hits and wire hits is used to determined 2D RecHit.
 * For layers where only strip or wire hits are present, pseudo 2D hits are formed.
 *
 * \author: Dominique Fortin - UCR
 *
 */
//---- Possible changes from Stoyan Stoynev - NU

#include <RecoLocalMuon/CSCRecHitD/src/CSCWireHit.h>
#include <RecoLocalMuon/CSCRecHitD/src/CSCStripHit.h>

#include <DataFormats/CSCRecHit/interface/CSCRecHit2D.h>
// Forward declaration of LocalPoint fails... due to typedef'ing?
#include <DataFormats/GeometryVector/interface/LocalPoint.h>

#include <FWCore/ParameterSet/interface/ParameterSet.h>

class CSCDetId;
class CSCLayer;
class CSCChamberSpecs;
class CSCLayerGeometry;
class CSCDBGains;
class CSCDBCrosstalk;
class CSCDBNoiseMatrix;
class CSCStripCrosstalk;
class CSCStripNoiseMatrix;
class CSCXonStrip_MatchGatti;

class CSCMake2DRecHit
{
 public:
  
  explicit CSCMake2DRecHit(const edm::ParameterSet& );
  
  ~CSCMake2DRecHit();
  
  /// Make 2D hits when have both wire and strip hit available in same layer
  CSCRecHit2D hitFromStripAndWire(const CSCDetId& id, const CSCLayer* layer, const CSCWireHit& wHit, const CSCStripHit& sHit);


  /// Test if rechit is in fiducial volume 
  bool isHitInFiducial( const CSCLayer* layer, const CSCRecHit2D& rh );

  /// Load in X-Talks and Noise Matrix
  void setCalibration( float GlobalGainAvg,
                       const CSCDBGains* gains,
                       const CSCDBCrosstalk* xtalk,
                       const CSCDBNoiseMatrix* noise ) {
    globalGainAvg = GlobalGainAvg;
    gains_ = gains;
    xtalk_ = xtalk;
    noise_ = noise;
  }
 
  const CSCLayer*         layer_;
  const CSCLayerGeometry* layergeom_;
  const CSCChamberSpecs*  specs_;
  CSCDetId                id_;  
  
 private:
  
  bool debug;
  bool useCalib;
  int stripWireDeltaTime;
  bool useGatti;
  float maxGattiChi2;

  /* Cache calibrations for current event
   *
   */
  float globalGainAvg;
  const CSCDBGains*       gains_;
  const CSCDBCrosstalk*   xtalk_;
  const CSCDBNoiseMatrix* noise_;

  CSCStripCrosstalk*       stripCrosstalk_; 
  CSCStripNoiseMatrix*     stripNoiseMatrix_;
  CSCXonStrip_MatchGatti* xMatchGatti_;  
};

#endif

