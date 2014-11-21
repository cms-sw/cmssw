#ifndef TrackRecoDeDx_DeDxHitInfoProducer_H
#define TrackRecoDeDx_DeDxHitInfoProducer_H
// user include files

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h" 

#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/TrackReco/interface/DeDxData.h"
#include "DataFormats/TrackReco/interface/TrackDeDxHits.h"
#include "DataFormats/TrackReco/interface/DeDxHit.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "RecoTracker/DeDx/interface/DeDxTools.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

#include "DataFormats/TrackReco/interface/DeDxHitInfo.h"

//
// class declaration
//

class DeDxHitInfoProducer : public edm::stream::EDProducer<> {
public:
  explicit DeDxHitInfoProducer(const edm::ParameterSet&);
  ~DeDxHitInfoProducer();

private:
  virtual void beginRun(edm::Run const& run, const edm::EventSetup&) override;
  virtual void produce(edm::Event&, const edm::EventSetup&) override;

  void   makeCalibrationMap(const TrackerGeometry& tkGeom);
  void   processHit(const TrackingRecHit* recHit, float trackMomentum, float& cosine, reco::DeDxHitInfo& hitDeDxInfo,  LocalPoint HitLocalPos);

  // ----------member data ---------------------------

  edm::EDGetTokenT<TrajTrackAssociationCollection>   m_trajTrackAssociationTag;
  edm::EDGetTokenT<reco::TrackCollection>  m_tracksTag;

  bool useTrajectory;
  bool usePixel;
  bool useStrip;
  float meVperADCPixel;
  float meVperADCStrip;

  unsigned int minTrackHits;
  float        minTrackPt;
  float        maxTrackEta;

  std::string                       m_calibrationPath;
  bool                              useCalibration;
  bool                              shapetest;

  std::vector< std::vector<float> > calibGains; 
  unsigned int m_off;
};

#endif

