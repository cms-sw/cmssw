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
class CSCRecoConditions;
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

  /// Pass pointer to conditions data onwards
  void setConditions( const CSCRecoConditions* reco );

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

  CSCXonStrip_MatchGatti* xMatchGatti_;  
};

#endif

