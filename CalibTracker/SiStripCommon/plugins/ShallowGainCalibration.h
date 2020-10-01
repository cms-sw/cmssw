#ifndef SHALLOW_GAINCALIBRATION_PRODUCER
#define SHALLOW_GAINCALIBRATION_PRODUCER

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "CalibTracker/SiStripCommon/interface/ShallowTools.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "CondFormats/SiStripObjects/interface/SiStripLorentzAngle.h"
#include "CondFormats/DataRecord/interface/SiStripLorentzAngleRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"
#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"

#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"

#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
#include "CalibTracker/Records/interface/SiStripGainRcd.h"

#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"

class ShallowGainCalibration : public edm::stream::EDProducer<> {
public:
  explicit ShallowGainCalibration(const edm::ParameterSet&);

private:
  const edm::EDGetTokenT<edm::View<reco::Track> > tracks_token_;
  const edm::EDGetTokenT<TrajTrackAssociationCollection> association_token_;

  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeometry_token_;
  const edm::ESGetToken<SiStripGain, SiStripGainRcd> gain_token_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> tkGeom_token_;

  std::string Suffix;
  std::string Prefix;

  void produce(edm::Event&, const edm::EventSetup&) override;
  bool isFarFromBorder(TrajectoryStateOnSurface* trajState, const uint32_t detid, const edm::EventSetup* iSetup);
  double thickness(DetId id);

  const TrackerGeometry* m_tracker;
  std::map<DetId, double> m_thicknessMap;
};
#endif
