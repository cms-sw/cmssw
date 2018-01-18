#ifndef SiPixelPhase1TrackClusters_h
#define SiPixelPhase1TrackClusters_h

#include "DQM/SiPixelPhase1Common/interface/SiPixelPhase1Base.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "DataFormats/SiPixelCluster/interface/SiPixelClusterShapeCache.h"

class SiPixelPhase1TrackClusters : public SiPixelPhase1Base {
enum {  
  ON_TRACK_CHARGE,
  ON_TRACK_SIZE,
  ON_TRACK_SHAPE,
  ON_TRACK_NCLUSTERS,
  ON_TRACK_POSITIONB,
  ON_TRACK_POSITIONF,
  DIGIS_HITMAP_ON_TRACK,

  NTRACKS,
  NTRACKS_INVOLUME,

  SIZE_VS_ETA_ON_TRACK_OUTER,
  SIZE_VS_ETA_ON_TRACK_INNER,
  ON_TRACK_CHARGE_OUTER,
  ON_TRACK_CHARGE_INNER,

  ON_TRACK_SHAPE_OUTER,
  ON_TRACK_SHAPE_INNER,
 
  ON_TRACK_SIZE_X_OUTER,
  ON_TRACK_SIZE_X_INNER,
  ON_TRACK_SIZE_X_F,
  ON_TRACK_SIZE_Y_OUTER,
  ON_TRACK_SIZE_Y_INNER,
  ON_TRACK_SIZE_Y_F,
    
  ON_TRACK_SIZE_XY_OUTER,
  ON_TRACK_SIZE_XY_INNER,
  ON_TRACK_SIZE_XY_F,
  CHARGE_VS_SIZE_ON_TRACK,

  ENUM_SIZE        
};

public:
  explicit SiPixelPhase1TrackClusters(const edm::ParameterSet& conf);
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  const bool applyVertexCut_;

  edm::EDGetTokenT<reco::TrackCollection> tracksToken_;
  edm::EDGetTokenT<reco::VertexCollection> offlinePrimaryVerticesToken_;
  edm::EDGetTokenT<SiPixelClusterShapeCache> pixelClusterShapeCacheToken_;
};


#endif