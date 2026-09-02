//
// Package:    MuonSimHitProducer
// Class:      MuonSimHitProducer
//
/**\class MuonSimHitProducer FastSimulation/MuonSimHitProducer/src/MuonSimHitProducer.cc

 Description:
    Fast simulation producer of Muon Sim Hits (to be used for realistic Muon reconstruction)

 Implementation:
     <Notes on implementation>

*/
//
// Original Author:  Martijn Mulders/Matthew Jones
//         Created:  Wed Jul 30 11:37:24 CET 2007
//         Working:  Fri Nov  9 09:39:33 CST 2007
//
// $Id: MuonSimHitProducer.cc,v 1.36 2011/10/07 08:25:42 aperrott Exp $
//
//

#include "DataFormats/GeometrySurface/interface/PlaneBuilder.h"
#include "DataFormats/GeometrySurface/interface/TangentPlane.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "FastSimulation/MaterialEffects/interface/EnergyLossSimulator.h"
#include "FastSimulation/MaterialEffects/interface/MaterialEffects.h"
#include "FastSimulation/MaterialEffects/interface/MultipleScatteringSimulator.h"
#include "FastSimulation/MaterialEffects/interface/MuonBremsstrahlungSimulator.h"
#include "FastSimulation/ParticlePropagator/interface/ParticlePropagator.h"
#include "FastSimulation/Utilities/interface/RandomEngineAndDistribution.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoMuon/MeasurementDet/interface/MuonDetLayerMeasurements.h"
#include "RecoMuon/Navigation/interface/DirectMuonNavigation.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"
#include "TrackingTools/GeomPropagators/interface/HelixArbitraryPlaneCrossing.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"

#include <cmath>

class MuonSimHitProducer : public edm::stream::EDProducer<edm::stream::WatchRuns> {
public:
  explicit MuonSimHitProducer(const edm::ParameterSet&);

private:
  static constexpr double c_cm_ns_ = 29.98;

  std::unique_ptr<MuonServiceProxy> theService_;
  Chi2MeasurementEstimator theEstimator_;

  const MagneticField* magfield_;
  const DTGeometry* dtGeom_;
  const CSCGeometry* cscGeom_;
  const RPCGeometry* rpcGeom_;
  const GEMGeometry* gemGeom_;
  const Propagator* propagatorWithMaterial_;
  std::unique_ptr<Propagator> propagatorWithoutMaterial_;
  bool enableGEM_;

  std::unique_ptr<MaterialEffects> theMaterialEffects_;

  void beginRun(edm::Run const& run, const edm::EventSetup& es) override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  void readParameters(const edm::ParameterSet&, const edm::ParameterSet&, const edm::ParameterSet&);

  // Parameters to emulate the muonSimHit association inefficiency due to delta's
  double kDT_;
  double fDT_;
  double kCSC_;
  double fCSC_;

  /// Simulate material effects in iron (dE/dx, multiple scattering)
  void applyMaterialEffects(TrajectoryStateOnSurface& tsosWithdEdx,
                            TrajectoryStateOnSurface& tsos,
                            double radPath,
                            RandomEngineAndDistribution const*,
                            HepPDT::ParticleDataTable const&);

  // ----------- parameters ----------------------------
  bool fullPattern_;
  bool doL1_, doL3_, doGL_;

  // tags
  edm::InputTag simMuonLabel_;
  edm::InputTag simVertexLabel_;

  // tokens
  edm::EDGetTokenT<std::vector<SimTrack>> simMuonToken_;
  edm::EDGetTokenT<std::vector<SimVertex>> simVertexToken_;

  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magneticFieldESToken_;
  const edm::ESGetToken<DTGeometry, MuonGeometryRecord> DTGeometryESToken_;
  const edm::ESGetToken<CSCGeometry, MuonGeometryRecord> CSCGeometryESToken_;
  const edm::ESGetToken<RPCGeometry, MuonGeometryRecord> RPCGeometryESToken_;
  edm::ESGetToken<GEMGeometry, MuonGeometryRecord> GEMGeometryESToken_;
  const edm::ESGetToken<HepPDT::ParticleDataTable, edm::DefaultRecord> particleDataTableESToken_;
};

//for debug only
//#define EDM_ML_DEBUG

//
// constructors and destructor
//
MuonSimHitProducer::MuonSimHitProducer(const edm::ParameterSet& iConfig)
    : theEstimator_(iConfig.getParameter<double>("Chi2EstimatorCut")),
      propagatorWithoutMaterial_(nullptr),
      enableGEM_(iConfig.getParameter<bool>("enableGEM")),
      magneticFieldESToken_(esConsumes<edm::Transition::BeginRun>()),
      DTGeometryESToken_(esConsumes<edm::Transition::BeginRun>(edm::ESInputTag("", "MisAligned"))),
      CSCGeometryESToken_(esConsumes<edm::Transition::BeginRun>(edm::ESInputTag("", "MisAligned"))),
      RPCGeometryESToken_(esConsumes<edm::Transition::BeginRun>()),
      particleDataTableESToken_(esConsumes()) {
  if (enableGEM_)
    GEMGeometryESToken_ = esConsumes<edm::Transition::BeginRun>();

  // Read relevant parameters
  readParameters(iConfig.getParameter<edm::ParameterSet>("MUONS"),
                 iConfig.getParameter<edm::ParameterSet>("TRACKS"),
                 iConfig.getParameter<edm::ParameterSet>("MaterialEffectsForMuons"));

  //
  //  register your products ... need to declare at least one possible product...
  //
  produces<edm::PSimHitContainer>("MuonCSCHits");
  produces<edm::PSimHitContainer>("MuonDTHits");
  produces<edm::PSimHitContainer>("MuonRPCHits");
  if (enableGEM_)
    produces<edm::PSimHitContainer>("MuonGEMHits");

  edm::ParameterSet serviceParameters = iConfig.getParameter<edm::ParameterSet>("ServiceParameters");
  theService_ = std::make_unique<MuonServiceProxy>(
      serviceParameters, consumesCollector(), MuonServiceProxy::UseEventSetupIn::Run);

  // consumes
  simMuonToken_ = consumes<std::vector<SimTrack>>(simMuonLabel_);
  simVertexToken_ = consumes<std::vector<SimVertex>>(simVertexLabel_);
}

// ---- method called once each job just before starting event loop ----
void MuonSimHitProducer::beginRun(edm::Run const& run, const edm::EventSetup& es) {
  //services
  magfield_ = &es.getData(magneticFieldESToken_);
  dtGeom_ = &es.getData(DTGeometryESToken_);
  cscGeom_ = &es.getData(CSCGeometryESToken_);
  rpcGeom_ = &es.getData(RPCGeometryESToken_);
  if (enableGEM_)
    gemGeom_ = &es.getData(GEMGeometryESToken_);

  bool duringEvent = false;
  theService_->update(es, duringEvent);

  // A few propagators
  propagatorWithMaterial_ = &(*(theService_->propagator("SteppingHelixPropagatorAny")));
  propagatorWithoutMaterial_.reset(propagatorWithMaterial_->clone());
  SteppingHelixPropagator* SHpropagator =
      dynamic_cast<SteppingHelixPropagator*>(propagatorWithoutMaterial_.get());  // Beuark!
  SHpropagator->setMaterialMode(true);                                           // switches OFF material effects;
}

//
// member functions
//

// ------------ method called to produce the data  ------------

void MuonSimHitProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto pdg = iSetup.getHandle(particleDataTableESToken_);

  RandomEngineAndDistribution random(iEvent.streamID());

  MuonPatternRecoDumper dumper;

  edm::Handle<std::vector<SimTrack>> simMuons;
  edm::Handle<std::vector<SimVertex>> simVertices;
  auto theCSCHits = std::make_unique<edm::PSimHitContainer>();
  auto theDTHits = std::make_unique<edm::PSimHitContainer>();
  auto theRPCHits = std::make_unique<edm::PSimHitContainer>();
  auto theGEMHits = std::make_unique<edm::PSimHitContainer>();

  DirectMuonNavigation navigation(theService_->detLayerGeometry());
  iEvent.getByToken(simMuonToken_, simMuons);
  iEvent.getByToken(simVertexToken_, simVertices);

  for (const auto& mySimTrack : *simMuons) {
    // Decaying hadrons are now in the list, and so are their muon daughter
    // Ignore the hadrons here.
    int pid = mySimTrack.type();
    if (abs(pid) != 13 && abs(pid) != 1000024)
      continue;
    double t0 = 0;
    GlobalPoint initialPosition;
    int ivert = mySimTrack.vertIndex();
    if (ivert >= 0) {
      t0 = (*simVertices)[ivert].position().t();
      initialPosition = GlobalPoint((*simVertices)[ivert].position().x(),
                                    (*simVertices)[ivert].position().y(),
                                    (*simVertices)[ivert].position().z());
    }
    //
    //  Presumably t0 has dimensions of cm if not mm?
    //  Convert to ns for internal calculations.
    //  I wonder where we should get c from?
    //
    double tof = t0 / c_cm_ns_;

#ifdef EDM_ML_DEBUG
    std::cout << " ===> MuonSimHitProducer::reconstruct() found SIMTRACK - pid = " << pid;
    std::cout << " : pT = " << mySimTrack.momentum().Pt() << ", eta = " << mySimTrack.momentum().Eta()
              << ", phi = " << mySimTrack.momentum().Phi() << std::endl;
#endif

    //
    //  Produce muons sim hits starting from undecayed simulated muons
    //

    GlobalPoint startingPosition(mySimTrack.trackerSurfacePosition().x(),
                                 mySimTrack.trackerSurfacePosition().y(),
                                 mySimTrack.trackerSurfacePosition().z());
    GlobalVector startingMomentum(mySimTrack.trackerSurfaceMomentum().x(),
                                  mySimTrack.trackerSurfaceMomentum().y(),
                                  mySimTrack.trackerSurfaceMomentum().z());
    //
    //  Crap... there's no time-of-flight to the trackerSurfacePosition()...
    //  So, this will be wrong when the curvature can't be neglected, but that
    //  will be rather seldom...  May as well ignore the mass too.
    //
    GlobalVector dtracker = startingPosition - initialPosition;
    tof += dtracker.mag() / c_cm_ns_;

#ifdef EDM_ML_DEBUG
    std::cout << " the Muon START position " << startingPosition << std::endl;
    std::cout << " the Muon START momentum " << startingMomentum << std::endl;
#endif

    //
    //  Some magic to define a TrajectoryStateOnSurface
    //
    PlaneBuilder pb;
    GlobalVector zAxis = startingMomentum.unit();
    GlobalVector yAxis(zAxis.y(), -zAxis.x(), 0);
    GlobalVector xAxis = yAxis.cross(zAxis);
    Surface::RotationType rot(xAxis, yAxis, zAxis);
    PlaneBuilder::ReturnType startingPlane = pb.plane(startingPosition, rot);
    GlobalTrajectoryParameters gtp(startingPosition, startingMomentum, (int)mySimTrack.charge(), magfield_);
    TrajectoryStateOnSurface startingState(gtp, *startingPlane);

    const std::vector<const DetLayer*>& navLayers =
        (std::abs(startingState.globalMomentum().eta()) > 4.5)
            ? navigation.compatibleEndcapLayers(*(startingState.freeState()), alongMomentum)
            : navigation.compatibleLayers(*(startingState.freeState()), alongMomentum);

    if (navLayers.empty())
      continue;

#ifdef EDM_ML_DEBUG
    std::cout << "Found " << navLayers.size() << " compatible DetLayers..." << std::endl;
#endif

    TrajectoryStateOnSurface propagatedState = startingState;
    for (unsigned int ilayer = 0; ilayer < navLayers.size(); ilayer++) {
#ifdef EDM_ML_DEBUG
      std::cout << "Propagating to layer " << ilayer << " " << dumper.dumpLayer(navLayers[ilayer]) << std::endl;
#endif

      const std::vector<DetWithState>& comps =
          navLayers[ilayer]->compatibleDets(propagatedState, *propagatorWithMaterial_, theEstimator_);
      if (comps.empty())
        continue;

#ifdef EDM_ML_DEBUG
      std::cout << "Propagating " << propagatedState << std::endl;
#endif

      // Starting momentum
      double pi = propagatedState.globalMomentum().mag();

      // Propagate with material effects (dE/dx average only)
      SteppingHelixStateInfo shsStart(*(propagatedState.freeTrajectoryState()));
      SteppingHelixStateInfo shsDest;
      ((const SteppingHelixPropagator*)propagatorWithMaterial_)
          ->propagate(shsStart, navLayers[ilayer]->surface(), shsDest);
      std::pair<TrajectoryStateOnSurface, double> next(shsDest.getStateOnSurface(navLayers[ilayer]->surface()),
                                                       shsDest.path());
      // No need to continue if there is no valid propagation available.
      // This happens rarely (~0.1% of ttbar events)
      if (!next.first.isValid())
        continue;
      // This is the estimate of the number of radiation lengths traversed,
      // together with the total path length
      double radPath = shsDest.radPath();
      double pathLength = next.second;

      // Now propagate without dE/dx (average)
      // [To add the dE/dx fluctuations to the actual dE/dx]
      std::pair<TrajectoryStateOnSurface, double> nextNoMaterial =
          propagatorWithoutMaterial_->propagateWithPath(propagatedState, navLayers[ilayer]->surface());

      // Update the propagated state
      propagatedState = next.first;
      double pf = propagatedState.globalMomentum().mag();

      // Insert dE/dx fluctuations and multiple scattering
      // Skip this step if nextNoMaterial.first is not valid
      // This happens rarely (~0.02% of ttbar events)
      if (theMaterialEffects_ && nextNoMaterial.first.isValid())
        applyMaterialEffects(propagatedState, nextNoMaterial.first, radPath, &random, *pdg);
      // Check that the 'shaken' propagatedState is still valid, otherwise continue
      if (!propagatedState.isValid())
        continue;
      // (No evidence that this ever happens)
      //
      //  Consider this... 1 GeV muon has a velocity that is only 0.5% slower than c...
      //  We probably can safely ignore the mass for anything that makes it out to the
      //  muon chambers.
      //
      double pavg = 0.5 * (pi + pf);
      double m2 = mySimTrack.momentum().M2();
      double rbeta = sqrt(1 + m2 / (pavg * pavg)) / c_cm_ns_;
      double dtof = pathLength * rbeta;
      // GEMDigitizer need the eloss information.
      // The muon mass negligible when we calculate the energya. So energy loss is assumed as the momentum difference.
      double eloss = pi - pf;

#ifdef EDM_ML_DEBUG
      std::cout << "Propagated to next surface... path length = " << pathLength << " cm, dTOF = " << dtof << " ns"
                << std::endl;
#endif

      tof += dtof;

      for (unsigned int icomp = 0; icomp < comps.size(); icomp++) {
        const GeomDet* gd = comps[icomp].first;
        if (gd->subDetector() == GeomDetEnumerators::DT) {
          DTChamberId id(gd->geographicalId());
          const DTChamber* chamber = dtGeom_->chamber(id);
          const std::vector<const DTSuperLayer*>& superlayer = chamber->superLayers();
          for (unsigned int isl = 0; isl < superlayer.size(); isl++) {
            const std::vector<const DTLayer*>& layer = superlayer[isl]->layers();
            for (unsigned int ilayer = 0; ilayer < layer.size(); ilayer++) {
              DTLayerId lid = layer[ilayer]->id();
#ifdef EDM_ML_DEBUG
              std::cout << "    Extrapolated to DT (" << lid.wheel() << "," << lid.station() << "," << lid.sector()
                        << "," << lid.superlayer() << "," << lid.layer() << ")" << std::endl;
#endif

              const GeomDetUnit* det = dtGeom_->idToDetUnit(lid);

              HelixArbitraryPlaneCrossing crossing(propagatedState.globalPosition().basicVector(),
                                                   propagatedState.globalMomentum().basicVector(),
                                                   propagatedState.transverseCurvature(),
                                                   anyDirection);
              const std::pair<bool, double>& path = crossing.pathLength(det->surface());
              if (!path.first)
                continue;
              LocalPoint lpos = det->toLocal(GlobalPoint(crossing.position(path.second)));
              if (!det->surface().bounds().inside(lpos))
                continue;
              const DTTopology& dtTopo = layer[ilayer]->specificTopology();
              int wire = dtTopo.channel(lpos);
              if (wire - dtTopo.firstChannel() < 0 || wire - dtTopo.lastChannel() > 0)
                continue;
              // no drift cell here (on the chamber edge or just outside)
              // this hit would otherwise be discarded downstream in the digitizer

              DTWireId wid(lid, wire);
              double thickness = det->surface().bounds().thickness();
              LocalVector lmom = det->toLocal(GlobalVector(crossing.direction(path.second)));
              lmom = lmom.unit() * propagatedState.localMomentum().mag();

              // Factor that takes into account the (rec)hits lost because of delta's, etc.:
              // (Not fully satisfactory patch, but it seems to work...)
              double pmu = lmom.mag();
              double theDTHitIneff = pmu > 0 ? exp(kDT_ * log(pmu) + fDT_) : 0.;
              if (random.flatShoot() < theDTHitIneff)
                continue;

              double pz = std::abs(lmom.z());
              LocalPoint entry = lpos - 0.5 * thickness * lmom / pz;
              LocalPoint exit = lpos + 0.5 * thickness * lmom / pz;
              double dtof = path.second * rbeta;
              int trkid = mySimTrack.trackId();
              unsigned int id = wid.rawId();
              short unsigned int processType = 2;
              theDTHits->emplace_back(
                  entry, exit, lmom.mag(), tof + dtof, eloss, pid, id, trkid, lmom.theta(), lmom.phi(), processType);
            }
          }
        } else if (gd->subDetector() == GeomDetEnumerators::CSC) {
          CSCDetId id(gd->geographicalId());
          const CSCChamber* chamber = cscGeom_->chamber(id);
          const std::vector<const CSCLayer*>& layers = chamber->layers();
          for (unsigned int ilayer = 0; ilayer < layers.size(); ilayer++) {
            CSCDetId lid = layers[ilayer]->id();

#ifdef EDM_ML_DEBUG
            std::cout << "    Extrapolated to CSC (" << lid.endcap() << "," << lid.ring() << "," << lid.station() << ","
                      << lid.layer() << ")" << std::endl;
#endif

            const GeomDetUnit* det = cscGeom_->idToDetUnit(lid);
            HelixArbitraryPlaneCrossing crossing(propagatedState.globalPosition().basicVector(),
                                                 propagatedState.globalMomentum().basicVector(),
                                                 propagatedState.transverseCurvature(),
                                                 anyDirection);
            const std::pair<bool, double>& path = crossing.pathLength(det->surface());
            if (!path.first)
              continue;
            LocalPoint lpos = det->toLocal(GlobalPoint(crossing.position(path.second)));
            // For CSCs the Bounds are for chamber frames not gas regions
            //      if ( ! det->surface().bounds().inside(lpos) ) continue;
            // New function knows where the 'active' volume is:
            const CSCLayerGeometry* laygeom = layers[ilayer]->geometry();
            if (!laygeom->inside(lpos))
              continue;
            //double thickness = laygeom->thickness(); gives number which is about 20 times too big
            double thickness = det->surface().bounds().thickness();  // this one works much better...
            LocalVector lmom = det->toLocal(GlobalVector(crossing.direction(path.second)));
            lmom = lmom.unit() * propagatedState.localMomentum().mag();

            // Factor that takes into account the (rec)hits lost because of delta's, etc.:
            // (Not fully satisfactory patch, but it seems to work...)
            double pmu = lmom.mag();
            double theCSCHitIneff = pmu > 0 ? exp(kCSC_ * log(pmu) + fCSC_) : 0.;
            // Take into account the different geometry in ME11:
            if (id.station() == 1 && id.ring() == 1)
              theCSCHitIneff = theCSCHitIneff * 0.442;
            if (random.flatShoot() < theCSCHitIneff)
              continue;

            double pz = std::abs(lmom.z());
            LocalPoint entry = lpos - 0.5 * thickness * lmom / pz;
            LocalPoint exit = lpos + 0.5 * thickness * lmom / pz;
            double dtof = path.second * rbeta;
            int trkid = mySimTrack.trackId();
            unsigned int id = lid.rawId();
            short unsigned int processType = 2;
            theCSCHits->emplace_back(
                entry, exit, lmom.mag(), tof + dtof, eloss, pid, id, trkid, lmom.theta(), lmom.phi(), processType);
          }
        } else if (gd->subDetector() == GeomDetEnumerators::RPCBarrel ||
                   gd->subDetector() == GeomDetEnumerators::RPCEndcap) {
          RPCDetId id(gd->geographicalId());
          const RPCChamber* chamber = rpcGeom_->chamber(id);
          const std::vector<const RPCRoll*>& roll = chamber->rolls();
          for (unsigned int iroll = 0; iroll < roll.size(); iroll++) {
            RPCDetId rid = roll[iroll]->id();

#ifdef EDM_ML_DEBUG
            std::cout << "    Extrapolated to RPC (" << rid.ring() << "," << rid.station() << "," << rid.sector() << ","
                      << rid.subsector() << "," << rid.layer() << "," << rid.roll() << ")" << std::endl;
#endif

            const GeomDetUnit* det = rpcGeom_->idToDetUnit(rid);
            HelixArbitraryPlaneCrossing crossing(propagatedState.globalPosition().basicVector(),
                                                 propagatedState.globalMomentum().basicVector(),
                                                 propagatedState.transverseCurvature(),
                                                 anyDirection);
            const std::pair<bool, double>& path = crossing.pathLength(det->surface());
            if (!path.first)
              continue;
            LocalPoint lpos = det->toLocal(GlobalPoint(crossing.position(path.second)));
            if (!det->surface().bounds().inside(lpos))
              continue;
            double thickness = det->surface().bounds().thickness();
            LocalVector lmom = det->toLocal(GlobalVector(crossing.direction(path.second)));
            lmom = lmom.unit() * propagatedState.localMomentum().mag();
            double pz = std::abs(lmom.z());
            LocalPoint entry = lpos - 0.5 * thickness * lmom / pz;
            LocalPoint exit = lpos + 0.5 * thickness * lmom / pz;
            double dtof = path.second * rbeta;
            int trkid = mySimTrack.trackId();
            unsigned int id = rid.rawId();
            short unsigned int processType = 2;
            theRPCHits->emplace_back(
                entry, exit, lmom.mag(), tof + dtof, eloss, pid, id, trkid, lmom.theta(), lmom.phi(), processType);
          }
        } else if (gd->subDetector() == GeomDetEnumerators::GEM and enableGEM_) {
          GEMDetId id(gd->geographicalId());
          const GEMChamber* chamber = gemGeom_->chamber(id);
          const std::vector<const GEMEtaPartition*>& etaPart = chamber->etaPartitions();
          for (unsigned int ieta = 0; ieta < etaPart.size(); ieta++) {
            GEMDetId rid = etaPart[ieta]->id();

#ifdef EDM_ML_DEBUG
            std::cout << "    Extrapolated to GEM (" << rid.ring() << "," << rid.station() << "," << rid.chamber()
                      << "," << rid.layer() << "," << rid.ieta() << ")" << std::endl;
#endif

            const GeomDetUnit* det = gemGeom_->idToDetUnit(rid);
            HelixArbitraryPlaneCrossing crossing(propagatedState.globalPosition().basicVector(),
                                                 propagatedState.globalMomentum().basicVector(),
                                                 propagatedState.transverseCurvature(),
                                                 anyDirection);
            const std::pair<bool, double>& path = crossing.pathLength(det->surface());
            if (!path.first)
              continue;
            LocalPoint lpos = det->toLocal(GlobalPoint(crossing.position(path.second)));
            if (!det->surface().bounds().inside(lpos))
              continue;
            double thickness = det->surface().bounds().thickness();
            LocalVector lmom = det->toLocal(GlobalVector(crossing.direction(path.second)));
            lmom = lmom.unit() * propagatedState.localMomentum().mag();
            double pz = std::abs(lmom.z());
            LocalPoint entry = lpos - 0.5 * thickness * lmom / pz;
            LocalPoint exit = lpos + 0.5 * thickness * lmom / pz;
            double dtof = path.second * rbeta;
            int trkid = mySimTrack.trackId();
            unsigned int id = rid.rawId();
            short unsigned int processType = 2;
            theGEMHits->emplace_back(
                entry, exit, lmom.mag(), tof + dtof, eloss, pid, id, trkid, lmom.theta(), lmom.phi(), processType);
          }
        } else {
          edm::LogWarning("FastSimulation/MuonSimHitProducer")
              << "Extrapolated to unknown subdetector '" << gd->subDetector();
        }
      }
    }
  }

  iEvent.put(std::move(theCSCHits), "MuonCSCHits");
  iEvent.put(std::move(theDTHits), "MuonDTHits");
  iEvent.put(std::move(theRPCHits), "MuonRPCHits");
  if (enableGEM_)
    iEvent.put(std::move(theGEMHits), "MuonGEMHits");
}

void MuonSimHitProducer::readParameters(const edm::ParameterSet& fastMuons,
                                        const edm::ParameterSet& fastTracks,
                                        const edm::ParameterSet& matEff) {
  // Muons
  const std::string& _simModuleLabel = fastMuons.getParameter<std::string>("simModuleLabel");
  const std::string& _simModuleProcess = fastMuons.getParameter<std::string>("simModuleProcess");
  simMuonLabel_ = edm::InputTag(_simModuleLabel, _simModuleProcess);
  simVertexLabel_ = edm::InputTag(_simModuleLabel);

  const std::vector<double>& simHitIneffDT_ = fastMuons.getParameter<std::vector<double>>("simHitDTIneffParameters");
  const std::vector<double>& simHitIneffCSC_ = fastMuons.getParameter<std::vector<double>>("simHitCSCIneffParameters");
  kDT_ = simHitIneffDT_[0];
  fDT_ = simHitIneffDT_[1];
  kCSC_ = simHitIneffCSC_[0];
  fCSC_ = simHitIneffCSC_[1];

  // Tracks
  fullPattern_ = fastTracks.getUntrackedParameter<bool>("FullPatternRecognition");

  // Material Effects
  theMaterialEffects_ = nullptr;
  if (matEff.getParameter<bool>("PairProduction") || matEff.getParameter<bool>("Bremsstrahlung") ||
      matEff.getParameter<bool>("MuonBremsstrahlung") || matEff.getParameter<bool>("EnergyLoss") ||
      matEff.getParameter<bool>("MultipleScattering"))
    theMaterialEffects_ = std::make_unique<MaterialEffects>(matEff);
}

void MuonSimHitProducer::applyMaterialEffects(TrajectoryStateOnSurface& tsosWithdEdx,
                                              TrajectoryStateOnSurface& tsos,
                                              double radPath,
                                              RandomEngineAndDistribution const* random,
                                              HepPDT::ParticleDataTable const& table) {
  // The energy loss simulator
  EnergyLossSimulator* energyLoss = theMaterialEffects_->energyLossSimulator();

  // The multiple scattering simulator
  MultipleScatteringSimulator* multipleScattering = theMaterialEffects_->multipleScatteringSimulator();

  // The Muon Bremsstrahlung simulator
  MuonBremsstrahlungSimulator* bremsstrahlung = theMaterialEffects_->muonBremsstrahlungSimulator();

  // Initialize the Particle position, momentum and energy
  const Surface& nextSurface = tsos.surface();
  GlobalPoint gPos = energyLoss ? tsos.globalPosition() : tsosWithdEdx.globalPosition();
  GlobalVector gMom = energyLoss ? tsos.globalMomentum() : tsosWithdEdx.globalMomentum();
  double mu2 = std::pow(0.1056583692, 2);
  double en = std::sqrt(gMom.mag2() + mu2);

  // And now create the Particle
  XYZTLorentzVector position(gPos.x(), gPos.y(), gPos.z(), 0.);
  XYZTLorentzVector momentum(gMom.x(), gMom.y(), gMom.z(), en);
  float charge = (float)(tsos.charge());
  ParticlePropagator theMuon(rawparticle::makeMuon(charge < 1., momentum, position), nullptr, nullptr, &table);

  // Recompute the energy loss to get the fluctuations
  if (energyLoss) {
    // Difference between with and without dE/dx (average only)
    // (for corrections once fluctuations are applied)
    GlobalPoint gPosWithdEdx = tsosWithdEdx.globalPosition();
    GlobalVector gMomWithdEdx = tsosWithdEdx.globalMomentum();
    double enWithdEdx = std::sqrt(gMomWithdEdx.mag2() + mu2);
    XYZTLorentzVector deltaPos(
        gPosWithdEdx.x() - gPos.x(), gPosWithdEdx.y() - gPos.y(), gPosWithdEdx.z() - gPos.z(), 0.);
    XYZTLorentzVector deltaMom(
        gMomWithdEdx.x() - gMom.x(), gMomWithdEdx.y() - gMom.y(), gMomWithdEdx.z() - gMom.z(), enWithdEdx - en);

    // Energy loss in iron (+ fluctuations)
    energyLoss->updateState(theMuon, radPath, random);

    // Correcting factors to account for slight differences in material descriptions
    // (Material description is more accurate in the stepping helix propagator)
    radPath *= -deltaMom.E() / energyLoss->mostLikelyLoss();
    double fac = energyLoss->deltaMom().E() / energyLoss->mostLikelyLoss();

    // Particle momentum & position after energy loss + fluctuation
    XYZTLorentzVector theNewMomentum = theMuon.particle().momentum() + energyLoss->deltaMom() + fac * deltaMom;
    XYZTLorentzVector theNewPosition = theMuon.particle().vertex() + fac * deltaPos;
    fac = (theNewMomentum.E() * theNewMomentum.E() - mu2) / theNewMomentum.Vect().Mag2();
    fac = fac > 0. ? std::sqrt(fac) : 1E-9;
    theMuon.particle().setMomentum(
        theNewMomentum.Px() * fac, theNewMomentum.Py() * fac, theNewMomentum.Pz() * fac, theNewMomentum.E());
    theMuon.particle().setVertex(theNewPosition);
  }

  // Does the actual multiple scattering
  if (multipleScattering) {
    // Pass the vector normal to the "next" surface
    GlobalVector normal = nextSurface.tangentPlane(tsos.globalPosition())->normalVector();
    multipleScattering->setNormalVector(normal);
    // Compute the amount of multiple scattering after a given path length
    multipleScattering->updateState(theMuon, radPath, random);
  }

  // Muon Bremsstrahlung
  if (bremsstrahlung) {
    // Compute the amount of Muon Bremsstrahlung after given path length
    bremsstrahlung->updateState(theMuon, radPath, random);
  }

  // Fill the propagated state
  GlobalPoint propagatedPosition(theMuon.particle().X(), theMuon.particle().Y(), theMuon.particle().Z());
  GlobalVector propagatedMomentum(theMuon.particle().Px(), theMuon.particle().Py(), theMuon.particle().Pz());
  GlobalTrajectoryParameters propagatedGtp(propagatedPosition, propagatedMomentum, (int)charge, magfield_);
  tsosWithdEdx = TrajectoryStateOnSurface(propagatedGtp, nextSurface);
}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(MuonSimHitProducer);
