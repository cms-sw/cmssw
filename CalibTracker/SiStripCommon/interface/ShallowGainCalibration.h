#ifndef SHALLOW_GAINCALIBRATION_PRODUCER
#define SHALLOW_GAINCALIBRATION_PRODUCER

#include "FWCore/Framework/interface/EDProducer.h"
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

#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackReco/interface/DeDxHit.h"
#include "DataFormats/TrackReco/interface/TrackDeDxHits.h"

#include <ext/hash_map>

class ShallowGainCalibration : public edm::EDProducer {
public:
  explicit ShallowGainCalibration(const edm::ParameterSet&);

private:
  const edm::EDGetTokenT<edm::View<reco::Track> > tracks_token_;
  const edm::EDGetTokenT<TrajTrackAssociationCollection> association_token_;

  std::string Suffix;
  std::string Prefix;

  void produce(edm::Event&, const edm::EventSetup&) override;
  //  virtual void beginJob(EventSetup const&);
  //  virtual void beginRun(Run&, EventSetup const&);
  bool IsFarFromBorder(TrajectoryStateOnSurface* trajState, const uint32_t detid, const edm::EventSetup* iSetup);
  double thickness(DetId id);

  const TrackerGeometry* m_tracker;
  std::map<DetId, double> m_thicknessMap;

  /*
  struct stAPVGain{int DetId; int APVId; double PreviousGain;};
  class isEqual{
      public:
              template <class T> bool operator () (const T& PseudoDetId1, const T& PseudoDetId2) { return PseudoDetId1==PseudoDetId2; }
  };
  std::vector<stAPVGain*> APVsCollOrdered;
  hash_map<unsigned int, stAPVGain*,  hash<unsigned int>, isEqual > APVsColl;
*/
};
#endif
