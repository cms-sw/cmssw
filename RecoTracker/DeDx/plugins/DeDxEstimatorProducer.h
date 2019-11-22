#ifndef TrackRecoDeDx_DeDxEstimatorProducer_H
#define TrackRecoDeDx_DeDxEstimatorProducer_H
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
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GluedGeomDet.h"

#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/TrackReco/interface/DeDxData.h"
#include "DataFormats/TrackReco/interface/TrackDeDxHits.h"
#include "DataFormats/TrackReco/interface/DeDxHit.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "RecoTracker/DeDx/interface/BaseDeDxEstimator.h"
#include "RecoTracker/DeDx/interface/GenericAverageDeDxEstimator.h"
#include "RecoTracker/DeDx/interface/TruncatedAverageDeDxEstimator.h"
#include "RecoTracker/DeDx/interface/GenericTruncatedAverageDeDxEstimator.h"
#include "RecoTracker/DeDx/interface/MedianDeDxEstimator.h"
#include "RecoTracker/DeDx/interface/UnbinnedFitDeDxEstimator.h"
#include "RecoTracker/DeDx/interface/ProductDeDxDiscriminator.h"
#include "RecoTracker/DeDx/interface/SmirnovDeDxDiscriminator.h"
#include "RecoTracker/DeDx/interface/ASmirnovDeDxDiscriminator.h"
#include "RecoTracker/DeDx/interface/BTagLikeDeDxDiscriminator.h"

#include "RecoTracker/DeDx/interface/DeDxTools.h"

#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

//
// class declaration
//

class DeDxEstimatorProducer : public edm::stream::EDProducer<> {
public:
  explicit DeDxEstimatorProducer(const edm::ParameterSet&);
  ~DeDxEstimatorProducer() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginRun(edm::Run const& run, const edm::EventSetup&) override;
  void produce(edm::Event&, const edm::EventSetup&) override;

  void makeCalibrationMap(const TrackerGeometry& tkGeom);
  void processHit(const TrackingRecHit* recHit,
                  float trackMomentum,
                  float& cosine,
                  reco::DeDxHitCollection& dedxHits,
                  int& NClusterSaturating);

  // ----------member data ---------------------------
  BaseDeDxEstimator* m_estimator;

  edm::EDGetTokenT<reco::TrackCollection> m_tracksTag;

  bool usePixel;
  bool useStrip;
  float meVperADCPixel;
  float meVperADCStrip;

  unsigned int MaxNrStrips;

  std::string m_calibrationPath;
  bool useCalibration;
  bool shapetest;

  std::vector<std::vector<float> > calibGains;
  unsigned int m_off;

  edm::ESHandle<TrackerGeometry> tkGeom;
};

#endif
