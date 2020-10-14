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

  LogVerbatim("MTDLayerDump") << "\n*** allBTLLayers(): " << std::fixed << std::setw(14) << geo->allBTLLayers().size();
  for (auto dl = geo->allBTLLayers().begin(); dl != geo->allBTLLayers().end(); ++dl) {
    LogVerbatim("MTDLayerDump") << "  " << static_cast<int>(dl - geo->allBTLLayers().begin()) << " " << dumpLayer(*dl);
  }

  LogVerbatim("MTDLayerDump") << "\n*** allETLLayers(): " << std::fixed << std::setw(14) << geo->allETLLayers().size();
  for (auto dl = geo->allETLLayers().begin(); dl != geo->allETLLayers().end(); ++dl) {
    LogVerbatim("MTDLayerDump") << "  " << static_cast<int>(dl - geo->allETLLayers().begin()) << " " << dumpLayer(*dl);
  }

  LogVerbatim("MTDLayerDump") << "\n*** allLayers(): " << std::fixed << std::setw(14) << geo->allLayers().size();
  for (auto dl = geo->allLayers().begin(); dl != geo->allLayers().end(); ++dl) {
    LogVerbatim("MTDLayerDump") << "  " << static_cast<int>(dl - geo->allLayers().begin()) << " " << dumpLayer(*dl);
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

  for (const auto& ilay : layers) {
    const MTDTrayBarrelLayer* layer = static_cast<const MTDTrayBarrelLayer*>(ilay);

    LogVerbatim("MTDLayerDump") << std::fixed << "\nBTL layer " << std::setw(4) << layer->subDetector()
                                << " rods = " << std::setw(14) << layer->rods().size() << " dets = " << std::setw(14)
                                << layer->basicComponents().size();

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
    LogVerbatim("MTDLayerDump") << "\ntestBTLLayers: at " << std::fixed << std::setw(14) << tsos.globalPosition()
                                << " R=" << std::setw(14) << tsos.globalPosition().perp() << " phi=" << std::setw(14)
                                << tsos.globalPosition().phi() << " Z=" << std::setw(14) << tsos.globalPosition().z()
                                << " p = " << std::setw(14) << tsos.globalMomentum();

    SteppingHelixPropagator prop(field, anyDirection);

    pair<bool, TrajectoryStateOnSurface> comp = layer->compatible(tsos, prop, *theEstimator);
    LogVerbatim("MTDLayerDump") << "is compatible: " << comp.first << " at: R=" << std::setw(14)
                                << comp.second.globalPosition().perp() << " phi=" << std::setw(14)
                                << comp.second.globalPosition().phi() << " Z=" << std::setw(14)
                                << comp.second.globalPosition().z();

    vector<DetLayer::DetWithState> compDets = layer->compatibleDets(tsos, prop, *theEstimator);
    if (compDets.size()) {
      LogVerbatim("MTDLayerDump") << "compatibleDets: " << std::setw(14) << compDets.size() << "\n"
                                  << "  final state pos: " << std::setw(14) << compDets.front().second.globalPosition()
                                  << "\n"
                                  << "  det         pos: " << std::setw(14) << compDets.front().first->position()
                                  << " id: " << std::hex
                                  << BTLDetId(compDets.front().first->geographicalId().rawId()).rawId() << std::dec
                                  << "\n"
                                  << "  distance " << std::setw(14)
                                  << (tsos.globalPosition() - compDets.front().first->position()).mag();
    } else {
      LogVerbatim("MTDLayerDump") << " ERROR : no compatible det found";
    }
  }
}

void MTDRecoGeometryAnalyzer::testETLLayers(const MTDDetLayerGeometry* geo, const MagneticField* field) {
  const vector<const DetLayer*>& layers = geo->allETLLayers();

  for (const auto& ilay : layers) {
    const MTDRingForwardDoubleLayer* layer = static_cast<const MTDRingForwardDoubleLayer*>(ilay);

    LogVerbatim("MTDLayerDump") << std::fixed << "\nETL layer " << std::setw(4) << layer->subDetector()
                                << " rings = " << std::setw(14) << layer->rings().size() << " dets = " << std::setw(14)
                                << layer->basicComponents().size() << " front dets = " << std::setw(14)
                                << layer->frontLayer()->basicComponents().size() << " back dets = " << std::setw(14)
                                << layer->backLayer()->basicComponents().size();

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
    LogVerbatim("MTDLayerDump") << "\ntestETLLayers: at " << std::setw(14) << tsos.globalPosition()
                                << " R=" << std::setw(14) << tsos.globalPosition().perp() << " phi=" << std::setw(14)
                                << tsos.globalPosition().phi() << " Z=" << std::setw(14) << tsos.globalPosition().z()
                                << " p = " << std::setw(14) << tsos.globalMomentum();

    SteppingHelixPropagator prop(field, anyDirection);

    pair<bool, TrajectoryStateOnSurface> comp = layer->compatible(tsos, prop, *theEstimator);
    LogVerbatim("MTDLayerDump") << "is compatible: " << comp.first << " at: R=" << std::setw(14)
                                << comp.second.globalPosition().perp() << " phi=" << std::setw(14)
                                << comp.second.globalPosition().phi() << " Z=" << std::setw(14)
                                << comp.second.globalPosition().z();

    vector<DetLayer::DetWithState> compDets = layer->compatibleDets(tsos, prop, *theEstimator);
    if (compDets.size()) {
      LogVerbatim("MTDLayerDump") << "compatibleDets: " << std::setw(14) << compDets.size() << "\n"
                                  << "  final state pos: " << std::setw(14) << compDets.front().second.globalPosition()
                                  << "\n"
                                  << "  det         pos: " << std::setw(14) << compDets.front().first->position()
                                  << " id: " << std::hex
                                  << ETLDetId(compDets.front().first->geographicalId().rawId()).rawId() << std::dec
                                  << "\n"
                                  << "  distance " << std::setw(14)
                                  << (tsos.globalPosition() - compDets.front().first->position()).mag();
    } else {
      if (layer->isCrack(gp)) {
        LogVerbatim("MTDLayerDump") << " MTD crack found ";
      } else {
        LogVerbatim("MTDLayerDump") << " ERROR : no compatible det found in MTD"
                                    << " at: R=" << std::setw(14) << gp.perp() << " phi= " << std::setw(14)
                                    << gp.phi().degrees() << " Z= " << std::setw(14) << gp.z();
      }
    }
  }
}

void MTDRecoGeometryAnalyzer::testETLLayersNew(const MTDDetLayerGeometry* geo, const MagneticField* field) {
  const vector<const DetLayer*>& layers = geo->allETLLayers();

  // dump of ETL layers structure

  for (const auto& ilay : layers) {
    const MTDSectorForwardDoubleLayer* layer = static_cast<const MTDSectorForwardDoubleLayer*>(ilay);

    LogVerbatim("MTDLayerDump") << std::fixed << "\nETL layer " << std::setw(4) << layer->subDetector()
                                << " at z = " << std::setw(14) << layer->surface().position().z()
                                << " sectors = " << std::setw(14) << layer->sectors().size()
                                << " dets = " << std::setw(14) << layer->basicComponents().size()
                                << " front dets = " << std::setw(14) << layer->frontLayer()->basicComponents().size()
                                << " back dets = " << std::setw(14) << layer->backLayer()->basicComponents().size();

    unsigned int isectInd(0);
    for (const auto& isector : layer->sectors()) {
      isectInd++;
      LogVerbatim("MTDLayerDump") << std::fixed << "\nSector " << std::setw(4) << isectInd << "\n" << (*isector);
      for (const auto& imod : isector->basicComponents()) {
        ETLDetId modId(imod->geographicalId().rawId());
        LogVerbatim("MTDLayerDump") << std::fixed << "ETLDetId " << modId.rawId() << " side = " << std::setw(4)
                                    << modId.mtdSide() << " Disc/Side/Sector = " << std::setw(4) << modId.nDisc() << " "
                                    << std::setw(4) << modId.discSide() << " " << std::setw(4) << modId.sector()
                                    << " mod/type = " << std::setw(4) << modId.module() << " " << std::setw(4)
                                    << modId.modType() << " pos = " << imod->position();
      }
    }
  }

  // test propagation through layers

  for (const auto& ilay : layers) {
    const MTDSectorForwardDoubleLayer* layer = static_cast<const MTDSectorForwardDoubleLayer*>(ilay);

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
    LogVerbatim("MTDLayerDump") << "\ntestETLLayers: at " << std::fixed << tsos.globalPosition()
                                << " R=" << std::setw(14) << tsos.globalPosition().perp() << " phi=" << std::setw(14)
                                << tsos.globalPosition().phi() << " Z=" << std::setw(14) << tsos.globalPosition().z()
                                << " p = " << tsos.globalMomentum();

    SteppingHelixPropagator prop(field, anyDirection);

    pair<bool, TrajectoryStateOnSurface> comp = layer->compatible(tsos, prop, *theEstimator);
    LogVerbatim("MTDLayerDump") << std::fixed << "is compatible: " << comp.first << " at: R=" << std::setw(14)
                                << comp.second.globalPosition().perp() << " phi=" << std::setw(14)
                                << comp.second.globalPosition().phi() << " Z=" << std::setw(14)
                                << comp.second.globalPosition().z();

    vector<DetLayer::DetWithState> compDets = layer->compatibleDets(tsos, prop, *theEstimator);
    if (compDets.size()) {
      LogVerbatim("MTDLayerDump")
          << std::fixed << "compatibleDets: " << std::setw(14) << compDets.size() << "\n"
          << "  final state pos: " << compDets.front().second.globalPosition() << "\n"
          << "  det         pos: " << compDets.front().first->position() << " id: " << std::hex
          << ETLDetId(compDets.front().first->geographicalId().rawId()).rawId() << std::dec << "\n"
          << "  distance " << std::setw(14)
          << (compDets.front().second.globalPosition() - compDets.front().first->position()).mag();
    } else {
      if (layer->isCrack(gp)) {
        LogVerbatim("MTDLayerDump") << " MTD crack found ";
      } else {
        LogVerbatim("MTDLayerDump") << " ERROR : no compatible det found in MTD"
                                    << " at: R=" << std::setw(14) << gp.perp() << " phi= " << std::setw(14)
                                    << gp.phi().degrees() << " Z= " << std::setw(14) << gp.z();
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
    output << std::fixed << " subdet = " << std::setw(4) << layer->subDetector() << " Barrel = " << layer->isBarrel()
           << " Forward = " << layer->isForward() << "  Cylinder of radius: " << std::setw(14) << bc->radius();
  } else if ((bd = dynamic_cast<const BoundDisk*>(sur))) {
    output << std::fixed << " subdet = " << std::setw(4) << layer->subDetector() << " Barrel = " << layer->isBarrel()
           << " Forward = " << layer->isForward() << "  Disk at: " << std::setw(14) << bd->position().z();
  }
  return output.str();
}

//define this as a plug-in
#include <FWCore/Framework/interface/MakerMacros.h>
DEFINE_FWK_MODULE(MTDRecoGeometryAnalyzer);
