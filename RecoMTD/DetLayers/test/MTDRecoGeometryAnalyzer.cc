#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoMTD/DetLayers/interface/MTDDetLayerGeometry.h"
#include "RecoMTD/Records/interface/MTDRecoGeometryRecord.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "Geometry/Records/interface/MTDTopologyRcd.h"
#include "Geometry/MTDNumberingBuilder/interface/MTDTopology.h"
#include "Geometry/MTDCommonData/interface/MTDTopologyMode.h"

#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"

#include "RecoMTD/DetLayers/interface/MTDTrayBarrelLayer.h"
#include "RecoMTD/DetLayers/interface/MTDDetTray.h"
#include "RecoMTD/DetLayers/interface/MTDRingForwardDoubleLayer.h"
#include "RecoMTD/DetLayers/interface/MTDDetRing.h"
#include "RecoMTD/DetLayers/interface/MTDSectorForwardDoubleLayer.h"
#include "RecoMTD/DetLayers/interface/MTDDetSector.h"

#include <DataFormats/MuonDetId/interface/CSCDetId.h>

#include <DataFormats/ForwardDetId/interface/BTLDetId.h>
#include <DataFormats/ForwardDetId/interface/ETLDetId.h>

#include <sstream>

#include "CLHEP/Random/RandFlat.h"

using namespace std;
using namespace edm;

class MTDRecoGeometryAnalyzer : public EDAnalyzer {
public:
  MTDRecoGeometryAnalyzer(const ParameterSet& pset);

  virtual void analyze(const Event& ev, const EventSetup& es);

  void testBTLLayers(const MTDDetLayerGeometry*, const MagneticField* field);
  void testETLLayers(const MTDDetLayerGeometry*, const MagneticField* field);
  void testETLLayersNew(const MTDDetLayerGeometry*, const MagneticField* field);

  string dumpLayer(const DetLayer* layer) const;

private:
  MeasurementEstimator* theEstimator;

  const edm::ESInputTag tag_;
  edm::ESGetToken<MTDDetLayerGeometry, MTDRecoGeometryRecord> geomToken_;
  edm::ESGetToken<MTDTopology, MTDTopologyRcd> mtdtopoToken_;
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magfieldToken_;
};

MTDRecoGeometryAnalyzer::MTDRecoGeometryAnalyzer(const ParameterSet& iConfig) : tag_(edm::ESInputTag{"", ""}) {
  geomToken_ = esConsumes<MTDDetLayerGeometry, MTDRecoGeometryRecord>(tag_);
  mtdtopoToken_ = esConsumes<MTDTopology, MTDTopologyRcd>(tag_);
  magfieldToken_ = esConsumes<MagneticField, IdealMagneticFieldRecord>(tag_);

  float theMaxChi2 = 25.;
  float theNSigma = 3.;
  theEstimator = new Chi2MeasurementEstimator(theMaxChi2, theNSigma);
}

void MTDRecoGeometryAnalyzer::analyze(const Event& ev, const EventSetup& es) {
  auto geo = es.getTransientHandle(geomToken_);
  auto mtdtopo = es.getTransientHandle(mtdtopoToken_);
  auto magfield = es.getTransientHandle(magfieldToken_);

  // Some printouts

  LogVerbatim("MTDLayerDump") << "\n*** allBTLLayers(): " << geo->allBTLLayers().size();
  for (auto dl = geo->allBTLLayers().begin(); dl != geo->allBTLLayers().end(); ++dl) {
    LogVerbatim("MTDLayerDump") << "  " << (int)(dl - geo->allBTLLayers().begin()) << " " << dumpLayer(*dl);
  }

  LogVerbatim("MTDLayerDump") << "\n*** allETLLayers(): " << geo->allETLLayers().size();
  for (auto dl = geo->allETLLayers().begin(); dl != geo->allETLLayers().end(); ++dl) {
    LogVerbatim("MTDLayerDump") << "  " << (int)(dl - geo->allETLLayers().begin()) << " " << dumpLayer(*dl);
  }

  LogVerbatim("MTDLayerDump") << "\n*** allLayers(): " << geo->allLayers().size();
  for (auto dl = geo->allLayers().begin(); dl != geo->allLayers().end(); ++dl) {
    LogVerbatim("MTDLayerDump") << "  " << (int)(dl - geo->allLayers().begin()) << " " << dumpLayer(*dl);
  }

  testBTLLayers(geo.product(), magfield.product());
  if (mtdtopo->getMTDTopologyMode() <= static_cast<int>(MTDTopologyMode::Mode::barphiflat)) {
    testETLLayers(geo.product(), magfield.product());
  } else {
    testETLLayersNew(geo.product(), magfield.product());
  }
}

void MTDRecoGeometryAnalyzer::testBTLLayers(const MTDDetLayerGeometry* geo, const MagneticField* field) {
  const vector<const DetLayer*>& layers = geo->allBTLLayers();

  for (auto ilay = layers.begin(); ilay != layers.end(); ++ilay) {
    const MTDTrayBarrelLayer* layer = (const MTDTrayBarrelLayer*)(*ilay);

    LogVerbatim("MTDLayerDump") << "\nBTL layer " << layer->subDetector() << " rods = " << layer->rods().size()
                                << " dets = " << layer->basicComponents().size();

    const BoundCylinder& cyl = layer->specificSurface();

    double halfZ = cyl.bounds().length() / 2.;

    // Generate a random point on the cylinder
    double aPhi = CLHEP::RandFlat::shoot(-Geom::pi(), Geom::pi());
    double aZ = CLHEP::RandFlat::shoot(-halfZ, halfZ);
    GlobalPoint gp(GlobalPoint::Cylindrical(cyl.radius(), aPhi, aZ));

    // Momentum: 10 GeV, straight from the origin
    GlobalVector gv(GlobalVector::Spherical(gp.theta(), aPhi, 10.));

    //FIXME: only negative charge
    int charge = -1;

    GlobalTrajectoryParameters gtp(gp, gv, charge, field);
    TrajectoryStateOnSurface tsos(gtp, cyl);
    LogVerbatim("MTDLayerDump") << "\ntestBTLLayers: at " << tsos.globalPosition()
                                << " R=" << tsos.globalPosition().perp() << " phi=" << tsos.globalPosition().phi()
                                << " Z=" << tsos.globalPosition().z() << " p = " << tsos.globalMomentum();

    SteppingHelixPropagator prop(field, anyDirection);

    pair<bool, TrajectoryStateOnSurface> comp = layer->compatible(tsos, prop, *theEstimator);
    LogVerbatim("MTDLayerDump") << "is compatible: " << comp.first << " at: R=" << comp.second.globalPosition().perp()
                                << " phi=" << comp.second.globalPosition().phi()
                                << " Z=" << comp.second.globalPosition().z();

    vector<DetLayer::DetWithState> compDets = layer->compatibleDets(tsos, prop, *theEstimator);
    if (compDets.size()) {
      LogVerbatim("MTDLayerDump") << "compatibleDets: " << compDets.size() << "\n"
                                  << "  final state pos: " << compDets.front().second.globalPosition() << "\n"
                                  << "  det         pos: " << compDets.front().first->position() << " id: " << std::hex
                                  << BTLDetId(compDets.front().first->geographicalId().rawId()).rawId() << std::dec
                                  << "\n"
                                  << "  distance "
                                  << (tsos.globalPosition() - compDets.front().first->position()).mag();
    } else {
      LogVerbatim("MTDLayerDump") << " ERROR : no compatible det found";
    }
  }
}

void MTDRecoGeometryAnalyzer::testETLLayers(const MTDDetLayerGeometry* geo, const MagneticField* field) {
  const vector<const DetLayer*>& layers = geo->allETLLayers();

  for (auto ilay = layers.begin(); ilay != layers.end(); ++ilay) {
    const MTDRingForwardDoubleLayer* layer = (const MTDRingForwardDoubleLayer*)(*ilay);

    LogVerbatim("MTDLayerDump") << "\nETL layer " << layer->subDetector() << " rings = " << layer->rings().size()
                                << " dets = " << layer->basicComponents().size()
                                << " front dets = " << layer->frontLayer()->basicComponents().size()
                                << " back dets = " << layer->backLayer()->basicComponents().size();

    const BoundDisk& disk = layer->specificSurface();

    // Generate a random point on the disk
    double aPhi = CLHEP::RandFlat::shoot(-Geom::pi(), Geom::pi());
    double aR = CLHEP::RandFlat::shoot(disk.innerRadius(), disk.outerRadius());
    GlobalPoint gp(GlobalPoint::Cylindrical(aR, aPhi, disk.position().z()));

    // Momentum: 10 GeV, straight from the origin
    GlobalVector gv(GlobalVector::Spherical(gp.theta(), aPhi, 10.));

    //FIXME: only negative charge
    int charge = -1;

    GlobalTrajectoryParameters gtp(gp, gv, charge, field);
    TrajectoryStateOnSurface tsos(gtp, disk);
    LogVerbatim("MTDLayerDump") << "\ntestETLLayers: at " << tsos.globalPosition()
                                << " R=" << tsos.globalPosition().perp() << " phi=" << tsos.globalPosition().phi()
                                << " Z=" << tsos.globalPosition().z() << " p = " << tsos.globalMomentum();

    SteppingHelixPropagator prop(field, anyDirection);

    pair<bool, TrajectoryStateOnSurface> comp = layer->compatible(tsos, prop, *theEstimator);
    LogVerbatim("MTDLayerDump") << "is compatible: " << comp.first << " at: R=" << comp.second.globalPosition().perp()
                                << " phi=" << comp.second.globalPosition().phi()
                                << " Z=" << comp.second.globalPosition().z();

    vector<DetLayer::DetWithState> compDets = layer->compatibleDets(tsos, prop, *theEstimator);
    if (compDets.size()) {
      LogVerbatim("MTDLayerDump") << "compatibleDets: " << compDets.size() << "\n"
                                  << "  final state pos: " << compDets.front().second.globalPosition() << "\n"
                                  << "  det         pos: " << compDets.front().first->position() << " id: " << std::hex
                                  << ETLDetId(compDets.front().first->geographicalId().rawId()).rawId() << std::dec
                                  << "\n"
                                  << "  distance "
                                  << (tsos.globalPosition() - compDets.front().first->position()).mag();
    } else {
      if (layer->isCrack(gp)) {
        LogVerbatim("MTDLayerDump") << " MTD crack found ";
      } else {
        LogVerbatim("MTDLayerDump") << " ERROR : no compatible det found in MTD"
                                    << " at: R=" << gp.perp() << " phi= " << gp.phi().degrees() << " Z= " << gp.z();
      }
    }
  }
}

void MTDRecoGeometryAnalyzer::testETLLayersNew(const MTDDetLayerGeometry* geo, const MagneticField* field) {
  const vector<const DetLayer*>& layers = geo->allETLLayers();

  // dump of ETL layers structure

  for (auto ilay = layers.begin(); ilay != layers.end(); ++ilay) {
    const MTDSectorForwardDoubleLayer* layer = (const MTDSectorForwardDoubleLayer*)(*ilay);

    LogVerbatim("MTDLayerDump") << "\nETL layer " << layer->subDetector()
                                << " at z = " << layer->surface().position().z()
                                << " sectors = " << layer->sectors().size()
                                << " dets = " << layer->basicComponents().size()
                                << " front dets = " << layer->frontLayer()->basicComponents().size()
                                << " back dets = " << layer->backLayer()->basicComponents().size();

    unsigned int isectInd(0);
    for (const auto& isector : layer->sectors()) {
      isectInd++;
      LogVerbatim("MTDLayerDump") << "\nSector " << isectInd << " pos = " << isector->specificSurface().position()
                                  << " rmin = " << isector->specificSurface().innerRadius()
                                  << " rmax = " << isector->specificSurface().outerRadius()
                                  << " phi ref = " << isector->specificSurface().position().phi()
                                  << " phi/2 = " << isector->specificSurface().phiHalfExtension();
      for (const auto& imod : isector->basicComponents()) {
        ETLDetId modId(imod->geographicalId().rawId());
        LogVerbatim("MTDLayerDump") << "ETLDetId " << modId.rawId() << " side = " << modId.mtdSide()
                                    << " Disc/Side/Sector = " << modId.nDisc() << " " << modId.discSide() << " "
                                    << modId.sector() << " mod/type = " << modId.module() << " " << modId.modType()
                                    << " pos = " << imod->position();
      }
    }
  }

  // test propagation through layers

  for (auto ilay = layers.begin(); ilay != layers.end(); ++ilay) {
    const MTDSectorForwardDoubleLayer* layer = (const MTDSectorForwardDoubleLayer*)(*ilay);

    const BoundDisk& disk = layer->specificSurface();

    // Generate a random point on the disk
    double aPhi = CLHEP::RandFlat::shoot(-Geom::pi(), Geom::pi());
    double aR = CLHEP::RandFlat::shoot(disk.innerRadius(), disk.outerRadius());
    GlobalPoint gp(GlobalPoint::Cylindrical(aR, aPhi, disk.position().z()));

    // Momentum: 10 GeV, straight from the origin
    GlobalVector gv(GlobalVector::Spherical(gp.theta(), aPhi, 10.));

    //FIXME: only negative charge
    int charge = -1;

    GlobalTrajectoryParameters gtp(gp, gv, charge, field);
    TrajectoryStateOnSurface tsos(gtp, disk);
    LogInfo("MTDLayerDump") << "\ntestETLLayers: at " << tsos.globalPosition() << " R=" << tsos.globalPosition().perp()
                            << " phi=" << tsos.globalPosition().phi() << " Z=" << tsos.globalPosition().z()
                            << " p = " << tsos.globalMomentum();

    SteppingHelixPropagator prop(field, anyDirection);

    pair<bool, TrajectoryStateOnSurface> comp = layer->compatible(tsos, prop, *theEstimator);
    LogInfo("MTDLayerDump") << "is compatible: " << comp.first << " at: R=" << comp.second.globalPosition().perp()
                            << " phi=" << comp.second.globalPosition().phi()
                            << " Z=" << comp.second.globalPosition().z();

    vector<DetLayer::DetWithState> compDets = layer->compatibleDets(tsos, prop, *theEstimator);
    if (compDets.size()) {
      LogInfo("MTDLayerDump") << "compatibleDets: " << compDets.size() << "\n"
                              << "  final state pos: " << compDets.front().second.globalPosition() << "\n"
                              << "  det         pos: " << compDets.front().first->position() << " id: " << std::hex
                              << ETLDetId(compDets.front().first->geographicalId().rawId()).rawId() << std::dec << "\n"
                              << "  distance " << (tsos.globalPosition() - compDets.front().first->position()).mag();
    } else {
      if (layer->isCrack(gp)) {
        LogInfo("MTDLayerDump") << " MTD crack found ";
      } else {
        LogInfo("MTDLayerDump") << " ERROR : no compatible det found in MTD"
                                << " at: R=" << gp.perp() << " phi= " << gp.phi().degrees() << " Z= " << gp.z();
      }
    }
  }
}

string MTDRecoGeometryAnalyzer::dumpLayer(const DetLayer* layer) const {
  stringstream output;

  const BoundSurface* sur = 0;
  const BoundCylinder* bc = 0;
  const BoundDisk* bd = 0;

  sur = &(layer->surface());
  if ((bc = dynamic_cast<const BoundCylinder*>(sur))) {
    output << " subdet = " << layer->subDetector() << " Barrel = " << layer->isBarrel()
           << " Forward = " << layer->isForward() << "  Cylinder of radius: " << bc->radius();
  } else if ((bd = dynamic_cast<const BoundDisk*>(sur))) {
    output << " subdet = " << layer->subDetector() << " Barrel = " << layer->isBarrel()
           << " Forward = " << layer->isForward() << "  Disk at: " << bd->position().z();
  }
  return output.str();
}

//define this as a plug-in
#include <FWCore/Framework/interface/MakerMacros.h>
DEFINE_FWK_MODULE(MTDRecoGeometryAnalyzer);
