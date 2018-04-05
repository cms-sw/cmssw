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
#include "RecoTracker/DeDx/interface/GenericTruncatedAverageDeDxEstimator.h"

//
// class declaration
//

class DeDxHitInfoProducer : public edm::stream::EDProducer<> {
public:
  explicit DeDxHitInfoProducer(const edm::ParameterSet&);
  ~DeDxHitInfoProducer() override;

private:
  void beginRun(edm::Run const& run, const edm::EventSetup&) override;
  void produce(edm::Event&, const edm::EventSetup&) override;

  void   makeCalibrationMap(const TrackerGeometry& tkGeom);
  void   processHit(const TrackingRecHit* recHit, const float trackMomentum, const float cosine, reco::DeDxHitInfo& hitDeDxInfo,  const LocalPoint& hitLocalPos);

  // ----------member data ---------------------------

  edm::EDGetTokenT<reco::TrackCollection>  m_tracksTag;

  const bool usePixel;
  const bool useStrip;
  const float MeVperADCPixel;
  const float MeVperADCStrip;

  const unsigned int minTrackHits;
  const float        minTrackPt;
  const float        minTrackPtPrescale;
  const float        maxTrackEta;

  const std::string                       m_calibrationPath;
  const bool                              useCalibration;
  const bool                              shapetest;

  const unsigned int lowPtTracksPrescalePass, lowPtTracksPrescaleFail;
  GenericTruncatedAverageDeDxEstimator lowPtTracksEstimator;
  const float lowPtTracksDeDxThreshold;

  std::vector< std::vector<float> > calibGains; 
  unsigned int m_off;

  edm::ESHandle<TrackerGeometry> tkGeom;

  uint64_t xorshift128p(uint64_t state[2]) {
      uint64_t x = state[0];
      uint64_t const y = state[1];
      state[0] = y;
      x ^= x << 23; // a
      state[1] = x ^ y ^ (x >> 17) ^ (y >> 26); // b, c
      return state[1] + y;
  }
};

#endif
