#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include <RecoTracker/MeasurementDet/interface/MeasurementTracker.h>
#include <RecoTracker/Record/interface/CkfComponentsRecord.h>
#include "TrackingTools/DetLayers/interface/NavigationSchool.h"
#include "RecoTracker/Record/interface/NavigationSchoolRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "MagneticField/VolumeGeometry/interface/MagVolumeOutsideValidity.h"
#include "DataFormats/GeometrySurface/interface/PlaneBuilder.h"

#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackPropagation/RungeKutta/interface/defaultRKPropagator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"

namespace {

  Surface::RotationType rotation(const GlobalVector& zDir) {
    GlobalVector zAxis = zDir.unit();
    GlobalVector yAxis(zAxis.y(), -zAxis.x(), 0);
    GlobalVector xAxis = yAxis.cross(zAxis);
    return Surface::RotationType(xAxis, yAxis, zAxis);
  }

}  // namespace

class MeasurementTrackerTest : public edm::EDAnalyzer {
public:
  explicit MeasurementTrackerTest(const edm::ParameterSet&);
  ~MeasurementTrackerTest() override;

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  std::string theMeasurementTrackerName;
  std::string theNavigationSchoolName;
};

MeasurementTrackerTest::MeasurementTrackerTest(const edm::ParameterSet& iConfig)
    : theMeasurementTrackerName(iConfig.getParameter<std::string>("measurementTracker")),
      theNavigationSchoolName(iConfig.getParameter<std::string>("navigationSchool")) {}

MeasurementTrackerTest::~MeasurementTrackerTest() {}

void MeasurementTrackerTest::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  //get the measurementtracker
  edm::ESHandle<MeasurementTracker> measurementTracker;
  edm::ESHandle<NavigationSchool> navSchool;

  iSetup.get<CkfComponentsRecord>().get(theMeasurementTrackerName, measurementTracker);
  iSetup.get<NavigationSchoolRecord>().get(theNavigationSchoolName, navSchool);

  auto const& geom = *(TrackerGeometry const*)(*measurementTracker).geomTracker();
  auto const& searchGeom = *(*measurementTracker).geometricSearchTracker();
  auto const& dus = geom.detUnits();

  auto firstBarrel = geom.offsetDU(GeomDetEnumerators::tkDetEnum[1]);
  auto firstForward = geom.offsetDU(GeomDetEnumerators::tkDetEnum[2]);

  std::cout << "number of dets " << dus.size() << std::endl;
  std::cout << "Bl/Fw loc " << firstBarrel << '/' << firstForward << std::endl;

  edm::ESHandle<MagneticField> magfield;
  iSetup.get<IdealMagneticFieldRecord>().get(magfield);

  edm::ESHandle<Propagator> propagatorHandle;
  iSetup.get<TrackingComponentsRecord>().get("PropagatorWithMaterial", propagatorHandle);
  auto const& ANprop = *propagatorHandle;

  // error (very very small)
  ROOT::Math::SMatrixIdentity id;
  AlgebraicSymMatrix55 C(id);
  C *= 1.e-16;

  CurvilinearTrajectoryError err(C);

  //use negative sigma=-3.0 in order to use a more conservative definition of isInside() for Bounds classes.
  Chi2MeasurementEstimator estimator(30., -3.0, 0.5, 2.0, 0.5, 1.e12);  // same as defauts....

  KFUpdator kfu;
  LocalError he(0.01 * 0.01, 0, 0.02 * 0.02);

  for (float tl = 0.1f; tl < 3.0f; tl += 0.5f) {
    float p = 1.0f;
    float phi = 0.1415f;
    float tanlmd = tl;  // 0.1f;
    auto sinth2 = 1.f / (1.f + tanlmd * tanlmd);
    auto sinth = std::sqrt(sinth2);
    auto costh = tanlmd * sinth;

    std::cout << tanlmd << ' ' << sinth << ' ' << costh << std::endl;

    GlobalVector startingMomentum(p * std::sin(phi) * sinth, p * std::cos(phi) * sinth, p * costh);
    float z = 0.1f;
    GlobalPoint startingPosition(0, 0, z);

    // make TSOS happy
    //Define starting plane
    PlaneBuilder pb;
    auto rot = rotation(startingMomentum);
    auto startingPlane = pb.plane(startingPosition, rot);

    GlobalVector moms[3] = {0.5f * startingMomentum, startingMomentum, 10.f * startingMomentum};

    for (const auto& mom : moms) {
      TrajectoryStateOnSurface startingStateP(
          GlobalTrajectoryParameters(startingPosition, mom, 1, magfield.product()), err, *startingPlane);
      auto tsos = startingStateP;

      // auto layer = searchGeom.idToLayer(dus[firstBarrel]->geographicalId());
      const DetLayer* layer = searchGeom.pixelBarrelLayers().front();
      {
        auto it = layer;
        std::cout << "first layer " << (it->isBarrel() ? " Barrel" : " Forward") << " layer " << it->seqNum()
                  << " SubDet " << it->subDetector() << std::endl;
      }
      auto const& detWithState = layer->compatibleDets(tsos, ANprop, estimator);
      if (detWithState.empty()) {
        std::cout << "no det on first layer" << std::endl;
        continue;
      }
      auto did = detWithState.front().first->geographicalId();
      std::cout << "arrived at " << int(did) << std::endl;
      tsos = detWithState.front().second;
      std::cout << tsos.globalPosition() << ' ' << tsos.localError().positionError() << std::endl;

      SiPixelRecHit::ClusterRef pref;
      SiPixelRecHit hitpx(tsos.localPosition(), he, 1., *detWithState.front().first, pref);
      tsos = kfu.update(tsos, hitpx);
      std::cout << tsos.globalPosition() << ' ' << tsos.localError().positionError() << std::endl;

      for (auto il = 1; il < 5; ++il) {
        if (!layer)
          break;
        auto const& compLayers = (*navSchool).nextLayers(*layer, *tsos.freeState(), alongMomentum);
        layer = nullptr;
        for (auto it : compLayers) {
          if (it->basicComponents().empty()) {
            //this should never happen. but better protect for it
            std::cout
                << "a detlayer with no components: I cannot figure out a DetId from this layer. please investigate."
                << std::endl;
            continue;
          }
          std::cout << il << (it->isBarrel() ? " Barrel" : " Forward") << " layer " << it->seqNum() << " SubDet "
                    << it->subDetector() << std::endl;
          auto const& detWithState = it->compatibleDets(tsos, ANprop, estimator);
          if (detWithState.empty()) {
            std::cout << "no det on this layer" << std::endl;
            continue;
          }
          layer = it;
          auto did = detWithState.front().first->geographicalId();
          std::cout << "arrived at " << int(did) << std::endl;
          tsos = detWithState.front().second;
          std::cout << tsos.globalPosition() << ' ' << tsos.localError().positionError() << std::endl;
        }
      }  // layer loop
    }    // loop on moms
  }      // loop  on tanLa
}

//define this as a plug-in
DEFINE_FWK_MODULE(MeasurementTrackerTest);
