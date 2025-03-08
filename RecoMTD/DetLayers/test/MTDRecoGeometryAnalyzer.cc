#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "RecoMTD/DetLayers/interface/MTDDetLayerGeometry.h"
#include "RecoMTD/Records/interface/MTDRecoGeometryRecord.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "Geometry/Records/interface/MTDTopologyRcd.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDTopology.h"
#include "Geometry/MTDCommonData/interface/MTDTopologyMode.h"

#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"

#include "RecoMTD/DetLayers/interface/MTDTrayBarrelLayer.h"
#include "RecoMTD/DetLayers/interface/MTDDetTray.h"
#include "RecoMTD/DetLayers/interface/MTDSectorForwardDoubleLayer.h"
#include "RecoMTD/DetLayers/interface/MTDDetSector.h"

#include <DataFormats/MuonDetId/interface/CSCDetId.h>

#include <DataFormats/ForwardDetId/interface/BTLDetId.h>
#include <DataFormats/ForwardDetId/interface/ETLDetId.h>

#include <memory>
#include <sstream>

#include "CLHEP/Random/RandFlat.h"
#include "DataFormats/Math/interface/angle.h"
#include "DataFormats/Math/interface/Rounding.h"

using namespace std;
using namespace edm;
using namespace angle_units;
using namespace cms_rounding;

class MTDRecoGeometryAnalyzer : public global::EDAnalyzer<> {
public:
  MTDRecoGeometryAnalyzer(const ParameterSet& pset);

  void analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const override;

  void testBTLLayers(const MTDDetLayerGeometry*, const MagneticField* field) const;
  void testETLLayersNew(const MTDDetLayerGeometry*, const MagneticField* field) const;

  string dumpLayer(const DetLayer* layer) const;

private:
  std::unique_ptr<MeasurementEstimator> theEstimator;

  inline std::string fround(const double in, const size_t prec) const {
    std::stringstream ss;
    ss << std::setprecision(prec) << std::fixed << std::setw(14) << roundIfNear0(in);
    return ss.str();
  }

  inline std::string fvecround(const GlobalPoint vecin, const size_t prec) const {
    std::stringstream ss;
    ss << std::setprecision(prec) << std::fixed << std::setw(14) << roundVecIfNear0(vecin);
    return ss.str();
  }

  inline std::string fvecround(const GlobalVector vecin, const size_t prec) const {
    std::stringstream ss;
    ss << std::setprecision(prec) << std::fixed << std::setw(14) << roundVecIfNear0(vecin);
    return ss.str();
  }

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
  theEstimator = std::make_unique<Chi2MeasurementEstimator>(theMaxChi2, theNSigma);
}

void MTDRecoGeometryAnalyzer::analyze(edm::StreamID, edm::Event const&, edm::EventSetup const& es) const {
  auto geo = es.getTransientHandle(geomToken_);
  auto mtdtopo = es.getTransientHandle(mtdtopoToken_);
  auto magfield = es.getTransientHandle(magfieldToken_);

  // Some printouts

  LogVerbatim("MTDLayerDump") << "\n*** allBTLLayers(): " << std::fixed << std::setw(14) << geo->allBTLLayers().size();
  for (auto dl = geo->allBTLLayers().begin(); dl != geo->allBTLLayers().end(); ++dl) {
    LogVerbatim("MTDLayerDumpFull") << "  " << static_cast<int>(dl - geo->allBTLLayers().begin()) << " "
                                    << dumpLayer(*dl);
    LogVerbatim("MTDLayerDump") << "  " << static_cast<int>(dl - geo->allBTLLayers().begin()) << " " << dumpLayer(*dl);
  }

  LogVerbatim("MTDLayerDump") << "\n*** allETLLayers(): " << std::fixed << std::setw(14) << geo->allETLLayers().size();
  for (auto dl = geo->allETLLayers().begin(); dl != geo->allETLLayers().end(); ++dl) {
    LogVerbatim("MTDLayerDumpFull") << "  " << static_cast<int>(dl - geo->allETLLayers().begin()) << " "
                                    << dumpLayer(*dl);
    LogVerbatim("MTDLayerDump") << "  " << static_cast<int>(dl - geo->allETLLayers().begin()) << " " << dumpLayer(*dl);
  }

  LogVerbatim("MTDLayerDumpFull") << "\n*** allForwardLayers(): " << std::fixed << std::setw(14)
                                  << geo->allForwardLayers().size();
  LogVerbatim("MTDLayerDump") << "\n*** allForwardLayers(): " << std::fixed << std::setw(14)
                              << geo->allForwardLayers().size();
  for (auto dl = geo->allForwardLayers().begin(); dl != geo->allForwardLayers().end(); ++dl) {
    LogVerbatim("MTDLayerDumpFull") << "  " << static_cast<int>(dl - geo->allForwardLayers().begin()) << " "
                                    << dumpLayer(*dl);
    LogVerbatim("MTDLayerDump") << "  " << static_cast<int>(dl - geo->allForwardLayers().begin()) << " "
                                << dumpLayer(*dl);
  }

  LogVerbatim("MTDLayerDumpFull") << "\n*** allBackwardLayers(): " << std::fixed << std::setw(14)
                                  << geo->allBackwardLayers().size();
  LogVerbatim("MTDLayerDump") << "\n*** allBackwardLayers(): " << std::fixed << std::setw(14)
                              << geo->allBackwardLayers().size();
  for (auto dl = geo->allBackwardLayers().begin(); dl != geo->allBackwardLayers().end(); ++dl) {
    LogVerbatim("MTDLayerDumpFull") << "  " << static_cast<int>(dl - geo->allBackwardLayers().begin()) << " "
                                    << dumpLayer(*dl);
    LogVerbatim("MTDLayerDump") << "  " << static_cast<int>(dl - geo->allBackwardLayers().begin()) << " "
                                << dumpLayer(*dl);
  }

  LogVerbatim("MTDLayerDumpFull") << "\n*** allLayers(): " << std::fixed << std::setw(14) << geo->allLayers().size();
  LogVerbatim("MTDLayerDump") << "\n*** allLayers(): " << std::fixed << std::setw(14) << geo->allLayers().size();
  for (auto dl = geo->allLayers().begin(); dl != geo->allLayers().end(); ++dl) {
    LogVerbatim("MTDLayerDumpFull") << "  " << static_cast<int>(dl - geo->allLayers().begin()) << " " << dumpLayer(*dl);
    LogVerbatim("MTDLayerDump") << "  " << static_cast<int>(dl - geo->allLayers().begin()) << " " << dumpLayer(*dl);
  }

  testBTLLayers(geo.product(), magfield.product());
  testETLLayersNew(geo.product(), magfield.product());
}

void MTDRecoGeometryAnalyzer::testBTLLayers(const MTDDetLayerGeometry* geo, const MagneticField* field) const {
  const vector<const DetLayer*>& layers = geo->allBTLLayers();

  for (const auto& ilay : layers) {
    const MTDTrayBarrelLayer* layer = static_cast<const MTDTrayBarrelLayer*>(ilay);

    LogVerbatim("MTDLayerDumpFull") << std::fixed << "\nBTL layer " << std::setw(4) << layer->subDetector()
                                    << " rods = " << std::setw(14) << layer->rods().size()
                                    << " dets = " << std::setw(14) << layer->basicComponents().size();
    LogVerbatim("MTDLayerDump") << std::fixed << "\nBTL layer " << std::setw(4) << layer->subDetector()
                                << " rods = " << std::setw(14) << layer->rods().size() << " dets = " << std::setw(14)
                                << layer->basicComponents().size();

    unsigned int irodInd(0);
    for (const auto& irod : layer->rods()) {
      irodInd++;
      LogVerbatim("MTDLayerDumpFull") << std::fixed << "\nRod " << irodInd
                                      << " dets = " << irod->basicComponents().size() << "\n";
      LogVerbatim("MTDLayerDump") << std::fixed << "\nRod " << irodInd << " dets = " << irod->basicComponents().size()
                                  << "\n";
      for (const auto& imod : irod->basicComponents()) {
        BTLDetId modId(imod->geographicalId().rawId());
        LogVerbatim("MTDLayerDumpFull") << std::fixed << "BTLDetId " << modId.rawId() << " side = " << std::setw(4)
                                        << modId.mtdSide() << " rod = " << modId.mtdRR()
                                        << " type/RU/mod = " << std::setw(1) << modId.modType() << "/" << std::setw(1)
                                        << modId.runit() << "/" << std::setw(2) << modId.module()
                                        << " R = " << fround(imod->position().perp(), 4)
                                        << " phi = " << fround(imod->position().phi(), 4)
                                        << " Z = " << fround(imod->position().z(), 4);
        LogVerbatim("MTDLayerDump") << std::fixed << "BTLDetId " << modId.rawId() << " side = " << std::setw(4)
                                    << modId.mtdSide() << " rod = " << modId.mtdRR()
                                    << " type/RU/mod = " << std::setw(1) << modId.modType() << "/" << std::setw(1)
                                    << modId.runit() << "/" << std::setw(2) << modId.module()
                                    << " R = " << fround(imod->position().perp(), 2)
                                    << " phi = " << fround(imod->position().phi(), 2)
                                    << " Z = " << fround(imod->position().z(), 2);
      }
    }

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
    LogVerbatim("MTDLayerDumpFull") << "\ntestBTLLayers: at " << fvecround(tsos.globalPosition(), 4)
                                    << " R=" << fround(tsos.globalPosition().perp(), 4)
                                    << " phi=" << fround(tsos.globalPosition().phi(), 4)
                                    << " Z=" << fround(tsos.globalPosition().z(), 4)
                                    << " p = " << fvecround(tsos.globalMomentum(), 4);
    LogVerbatim("MTDLayerDump") << "\ntestBTLLayers: at " << fvecround(tsos.globalPosition(), 2)
                                << " R=" << fround(tsos.globalPosition().perp(), 2)
                                << " phi=" << fround(tsos.globalPosition().phi(), 2)
                                << " Z=" << fround(tsos.globalPosition().z(), 2)
                                << " p = " << fvecround(tsos.globalMomentum(), 2);

    SteppingHelixPropagator prop(field, anyDirection);

    pair<bool, TrajectoryStateOnSurface> comp = layer->compatible(tsos, prop, *theEstimator);
    LogVerbatim("MTDLayerDumpFull") << "is compatible: " << comp.first
                                    << " at: R=" << fround(comp.second.globalPosition().perp(), 4)
                                    << " phi=" << fround(comp.second.globalPosition().phi(), 4)
                                    << " Z=" << fround(comp.second.globalPosition().z(), 4);
    LogVerbatim("MTDLayerDump") << "is compatible: " << comp.first
                                << " at: R=" << fround(comp.second.globalPosition().perp(), 2)
                                << " phi=" << fround(comp.second.globalPosition().phi(), 2)
                                << " Z=" << fround(comp.second.globalPosition().z(), 2);

    vector<DetLayer::DetWithState> compDets = layer->compatibleDets(tsos, prop, *theEstimator);
    if (!compDets.empty()) {
      LogVerbatim("MTDLayerDumpFull") << "compatibleDets: " << std::setw(14) << compDets.size() << "\n"
                                      << "  final state pos: " << fvecround(compDets.front().second.globalPosition(), 4)
                                      << "\n"
                                      << "  det         pos: " << fvecround(compDets.front().first->position(), 4)
                                      << " id: " << std::hex
                                      << BTLDetId(compDets.front().first->geographicalId().rawId()).rawId() << std::dec
                                      << "\n"
                                      << "  distance "
                                      << fround((tsos.globalPosition() - compDets.front().first->position()).mag(), 2);
      LogVerbatim("MTDLayerDump") << "compatibleDets: " << std::setw(14) << compDets.size() << "\n"
                                  << "  final state pos: " << fvecround(compDets.front().second.globalPosition(), 2)
                                  << "\n"
                                  << "  det         pos: " << fvecround(compDets.front().first->position(), 2)
                                  << " id: " << std::hex
                                  << BTLDetId(compDets.front().first->geographicalId().rawId()).rawId() << std::dec
                                  << "\n"
                                  << "  distance "
                                  << fround((tsos.globalPosition() - compDets.front().first->position()).mag(), 2);
    } else {
      LogVerbatim("MTDLayerDumpFull") << " ERROR : no compatible det found";
      LogVerbatim("MTDLayerDump") << " ERROR : no compatible det found";
    }

    // scan in phi at the given z
    LogVerbatim("MTDLayerDumpFull") << "\nBTL phi scan at Z = " << fround(aZ, 4) << "\n";
    LogVerbatim("MTDLayerDump") << "\nBTL phi scan at Z = " << fround(aZ, 2) << "\n";
    aPhi = (int)(-piRadians * 1000) / 1000.;
    double dPhi = 0.005;
    uint32_t nTot(0);
    uint32_t nComp(0);
    while (aPhi <= piRadians) {
      nTot++;
      GlobalPoint gp(GlobalPoint::Cylindrical(cyl.radius(), aPhi, aZ));
      GlobalVector gv(GlobalVector::Spherical(gp.theta(), aPhi, 10.));
      GlobalTrajectoryParameters gtp(gp, gv, charge, field);
      TrajectoryStateOnSurface tsos(gtp, cyl);
      SteppingHelixPropagator prop(field, anyDirection);
      vector<DetLayer::DetWithState> compDets = layer->compatibleDets(tsos, prop, *theEstimator);
      std::stringstream ss;
      if (!compDets.empty()) {
        nComp++;
        for (const auto& dets : compDets) {
          ss << " " << BTLDetId(dets.first->geographicalId().rawId()).rawId();
        }
      }
      LogVerbatim("MTDLayerDumpFull") << "BTL scan at phi = " << std::fixed << std::setw(5) << aPhi
                                      << " compatible dets = " << std::setw(14) << compDets.size() << ss.str();
      LogVerbatim("MTDLayerDump") << "BTL scan at phi = " << std::fixed << std::setw(5) << aPhi
                                  << " compatible dets = " << std::setw(14) << compDets.size() << ss.str();
      aPhi += dPhi;
    }
    LogVerbatim("MTDLayerDumpFull") << "\nBTL scan total points = " << nTot << " compatible = " << nComp
                                    << " fraction = " << double(nComp) / double(nTot);
    LogVerbatim("MTDLayerDump") << "\nBTL scan total points = " << nTot << " compatible = " << nComp
                                << " fraction = " << double(nComp) / double(nTot);
  }
}

void MTDRecoGeometryAnalyzer::testETLLayersNew(const MTDDetLayerGeometry* geo, const MagneticField* field) const {
  const vector<const DetLayer*>& layers = geo->allETLLayers();

  // dump of ETL layers structure

  for (const auto& ilay : layers) {
    const MTDSectorForwardDoubleLayer* layer = static_cast<const MTDSectorForwardDoubleLayer*>(ilay);

    LogVerbatim("MTDLayerDumpFull") << std::fixed << "\nETL layer " << std::setw(4) << layer->subDetector()
                                    << " at z = " << fround(layer->surface().position().z(), 4)
                                    << " sectors = " << std::setw(14) << layer->sectors().size()
                                    << " dets = " << std::setw(14) << layer->basicComponents().size()
                                    << " front dets = " << std::setw(14)
                                    << layer->frontLayer()->basicComponents().size() << " back dets = " << std::setw(14)
                                    << layer->backLayer()->basicComponents().size();
    LogVerbatim("MTDLayerDump") << std::fixed << "\nETL layer " << std::setw(4) << layer->subDetector()
                                << " at z = " << fround(layer->surface().position().z(), 2)
                                << " sectors = " << std::setw(14) << layer->sectors().size()
                                << " dets = " << std::setw(14) << layer->basicComponents().size()
                                << " front dets = " << std::setw(14) << layer->frontLayer()->basicComponents().size()
                                << " back dets = " << std::setw(14) << layer->backLayer()->basicComponents().size();

    unsigned int isectInd(0);
    for (const auto& isector : layer->sectors()) {
      isectInd++;
      LogVerbatim("MTDLayerDumpFull") << std::fixed << "\nSector " << std::setw(4) << isectInd << "\n" << (*isector);
      LogVerbatim("MTDLayerDump") << std::fixed << "\nSector " << std::setw(4) << isectInd << "\n" << (*isector);
      for (const auto& imod : isector->basicComponents()) {
        ETLDetId modId(imod->geographicalId().rawId());
        LogVerbatim("MTDLayerDumpFull") << std::fixed << "ETLDetId " << modId.rawId() << " side = " << std::setw(4)
                                        << modId.mtdSide() << " Disc/Side/Sector = " << std::setw(4) << modId.nDisc()
                                        << " " << std::setw(4) << modId.discSide() << " " << std::setw(4)
                                        << modId.sector() << " mod/type/sensor = " << std::setw(4) << modId.module()
                                        << " " << std::setw(4) << modId.modType() << " " << std::setw(4)
                                        << modId.sensor() << " pos = " << fvecround(imod->position(), 4);
        LogVerbatim("MTDLayerDump") << std::fixed << "ETLDetId " << modId.rawId() << " side = " << std::setw(4)
                                    << modId.mtdSide() << " Disc/Side/Sector = " << std::setw(4) << modId.nDisc() << " "
                                    << std::setw(4) << modId.discSide() << " " << std::setw(4) << modId.sector()
                                    << " mod/type/sensor = " << std::setw(4) << modId.module() << " " << std::setw(4)
                                    << modId.modType() << " " << std::setw(4) << modId.sensor()
                                    << " pos = " << fvecround(imod->position(), 2);
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
    LogVerbatim("MTDLayerDumpFull") << "\ntestETLLayers: at " << fvecround(tsos.globalPosition(), 4)
                                    << " R=" << fround(tsos.globalPosition().perp(), 4)
                                    << " phi=" << fround(tsos.globalPosition().phi(), 4)
                                    << " Z=" << fround(tsos.globalPosition().z(), 4)
                                    << " p = " << fvecround(tsos.globalMomentum(), 4);
    LogVerbatim("MTDLayerDump") << "\ntestETLLayers: at " << fvecround(tsos.globalPosition(), 2)
                                << " R=" << fround(tsos.globalPosition().perp(), 2)
                                << " phi=" << fround(tsos.globalPosition().phi(), 2)
                                << " Z=" << fround(tsos.globalPosition().z(), 2)
                                << " p = " << fvecround(tsos.globalMomentum(), 2);

    SteppingHelixPropagator prop(field, anyDirection);

    pair<bool, TrajectoryStateOnSurface> comp = layer->compatible(tsos, prop, *theEstimator);
    LogVerbatim("MTDLayerDumpFull") << std::fixed << "is compatible: " << comp.first
                                    << " at: R=" << fround(comp.second.globalPosition().perp(), 4)
                                    << " phi=" << fround(comp.second.globalPosition().phi(), 4)
                                    << " Z=" << fround(comp.second.globalPosition().z(), 4);
    LogVerbatim("MTDLayerDump") << std::fixed << "is compatible: " << comp.first
                                << " at: R=" << fround(comp.second.globalPosition().perp(), 2)
                                << " phi=" << fround(comp.second.globalPosition().phi(), 2)
                                << " Z=" << fround(comp.second.globalPosition().z(), 2);

    vector<DetLayer::DetWithState> compDets = layer->compatibleDets(tsos, prop, *theEstimator);
    if (!compDets.empty()) {
      LogVerbatim("MTDLayerDumpFull")
          << std::fixed << "compatibleDets: " << std::setw(14) << compDets.size() << "\n"
          << "  final state pos: " << fvecround(compDets.front().second.globalPosition(), 4) << "\n"
          << "  det         pos: " << fvecround(compDets.front().first->position(), 4) << " id: " << std::hex
          << ETLDetId(compDets.front().first->geographicalId().rawId()).rawId() << std::dec << "\n"
          << "  distance "
          << fround((compDets.front().second.globalPosition() - compDets.front().first->position()).mag(), 4);
      LogVerbatim("MTDLayerDump")
          << std::fixed << "compatibleDets: " << std::setw(14) << compDets.size() << "\n"
          << "  final state pos: " << fvecround(compDets.front().second.globalPosition(), 2) << "\n"
          << "  det         pos: " << fvecround(compDets.front().first->position(), 2) << " id: " << std::hex
          << ETLDetId(compDets.front().first->geographicalId().rawId()).rawId() << std::dec << "\n"
          << "  distance "
          << fround((compDets.front().second.globalPosition() - compDets.front().first->position()).mag(), 2);
    } else {
      if (layer->isCrack(gp)) {
        LogVerbatim("MTDLayerDumpFull") << " MTD crack found ";
        LogVerbatim("MTDLayerDump") << " MTD crack found ";
      } else {
        LogVerbatim("MTDLayerDumpFull") << " ERROR : no compatible det found in MTD"
                                        << " at: R=" << fround(gp.perp(), 4)
                                        << " phi= " << fround(gp.phi().degrees(), 4) << " Z= " << fround(gp.z(), 4);
        LogVerbatim("MTDLayerDump") << " ERROR : no compatible det found in MTD"
                                    << " at: R=" << fround(gp.perp(), 4) << " phi= " << fround(gp.phi().degrees(), 2)
                                    << " Z= " << fround(gp.z(), 2);
      }
    }
  }
}

string MTDRecoGeometryAnalyzer::dumpLayer(const DetLayer* layer) const {
  stringstream output;

  const BoundSurface* sur = nullptr;
  const BoundCylinder* bc = nullptr;
  const BoundDisk* bd = nullptr;

  sur = &(layer->surface());
  if ((bc = dynamic_cast<const BoundCylinder*>(sur))) {
    output << std::fixed << " subdet = " << std::setw(4) << layer->subDetector() << " Barrel = " << layer->isBarrel()
           << " Forward = " << layer->isForward() << "  Cylinder of radius: " << fround(bc->radius(), 4);
  } else if ((bd = dynamic_cast<const BoundDisk*>(sur))) {
    output << std::fixed << " subdet = " << std::setw(4) << layer->subDetector() << " Barrel = " << layer->isBarrel()
           << " Forward = " << layer->isForward() << "  Disk at: " << fround(bd->position().z(), 4);
  }
  return output.str();
}

//define this as a plug-in
#include <FWCore/Framework/interface/MakerMacros.h>
DEFINE_FWK_MODULE(MTDRecoGeometryAnalyzer);
