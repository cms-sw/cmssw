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

// CMSSW headers
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/MakerMacros.h"

// Fast Simulation headers
#include "FastSimulation/Utilities/interface/RandomEngineAndDistribution.h"
#include "FastSimulation/MuonSimHitProducer/interface/MuonSimHitProducer.h"
#include "FastSimulation/MaterialEffects/interface/MaterialEffects.h"
#include "FastSimulation/MaterialEffects/interface/MultipleScatteringSimulator.h"
#include "FastSimulation/MaterialEffects/interface/EnergyLossSimulator.h"
#include "FastSimulation/ParticlePropagator/interface/ParticlePropagator.h"
#include "FastSimulation/MaterialEffects/interface/MuonBremsstrahlungSimulator.h"

// SimTrack
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

// STL headers
#include <vector>
#include <iostream>

// RecoMuon headers
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "RecoMuon/Navigation/interface/DirectMuonNavigation.h"
#include "RecoMuon/MeasurementDet/interface/MuonDetLayerMeasurements.h"
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"

// Tracking Tools
#include "TrackingTools/GeomPropagators/interface/HelixArbitraryPlaneCrossing.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"

// Data Formats
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "DataFormats/GeometrySurface/interface/PlaneBuilder.h"
#include "DataFormats/GeometrySurface/interface/TangentPlane.h"

////////////////////////////////////////////////////////////////////////////
// Geometry, Magnetic Field
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

////////////////////// Now find detector IDs:

// #include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"

//for debug only
//#define FAMOS_DEBUG

//
// constructors and destructor
//
MuonSimHitProducer::MuonSimHitProducer(const edm::ParameterSet& iConfig)
    : theEstimator(iConfig.getParameter<double>("Chi2EstimatorCut")), propagatorWithoutMaterial(nullptr) {
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

  edm::ParameterSet serviceParameters = iConfig.getParameter<edm::ParameterSet>("ServiceParameters");
  theService = new MuonServiceProxy(serviceParameters, consumesCollector(), MuonServiceProxy::UseEventSetupIn::Run);

  // consumes
  simMuonToken = consumes<std::vector<SimTrack> >(simMuonLabel);
  simVertexToken = consumes<std::vector<SimVertex> >(simVertexLabel);
}

// ---- method called once each job just before starting event loop ----
void MuonSimHitProducer::beginRun(edm::Run const& run, const edm::EventSetup& es) {
  //services
  edm::ESHandle<MagneticField> magField;
  edm::ESHandle<DTGeometry> dtGeometry;
  edm::ESHandle<CSCGeometry> cscGeometry;
  edm::ESHandle<RPCGeometry> rpcGeometry;
  edm::ESHandle<Propagator> propagator;

  es.get<IdealMagneticFieldRecord>().get(magField);
  es.get<MuonGeometryRecord>().get("MisAligned", dtGeometry);
  es.get<MuonGeometryRecord>().get("MisAligned", cscGeometry);
  es.get<MuonGeometryRecord>().get(rpcGeometry);

  magfield = &(*magField);
  dtGeom = &(*dtGeometry);
  cscGeom = &(*cscGeometry);
  rpcGeom = &(*rpcGeometry);

  bool duringEvent = false;
  theService->update(es, duringEvent);

  // A few propagators
  propagatorWithMaterial = &(*(theService->propagator("SteppingHelixPropagatorAny")));
  propagatorWithoutMaterial = propagatorWithMaterial->clone();
  SteppingHelixPropagator* SHpropagator = dynamic_cast<SteppingHelixPropagator*>(propagatorWithoutMaterial);  // Beuark!
  SHpropagator->setMaterialMode(true);  // switches OFF material effects;
}

MuonSimHitProducer::~MuonSimHitProducer() {
  // do anything here that needs to be done at destruction time
  // (e.g. close files, deallocate resources etc.)

  if (theMaterialEffects)
    delete theMaterialEffects;
  if (propagatorWithoutMaterial)
    delete propagatorWithoutMaterial;
}

//
// member functions
//

// ------------ method called to produce the data  ------------

void MuonSimHitProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // using namespace edm;
  // using namespace std;
  edm::ESHandle<HepPDT::ParticleDataTable> pdg;
  iSetup.getData(pdg);

  RandomEngineAndDistribution random(iEvent.streamID());

  MuonPatternRecoDumper dumper;

  edm::Handle<std::vector<SimTrack> > simMuons;
  edm::Handle<std::vector<SimVertex> > simVertices;
  std::vector<PSimHit> theCSCHits;
  std::vector<PSimHit> theDTHits;
  std::vector<PSimHit> theRPCHits;

  DirectMuonNavigation navigation(theService->detLayerGeometry());
  iEvent.getByToken(simMuonToken, simMuons);
  iEvent.getByToken(simVertexToken, simVertices);

  for (unsigned int itrk = 0; itrk < simMuons->size(); itrk++) {
    const SimTrack& mySimTrack = (*simMuons)[itrk];
    math::XYZTLorentzVector mySimP4(
        mySimTrack.momentum().x(), mySimTrack.momentum().y(), mySimTrack.momentum().z(), mySimTrack.momentum().t());

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
      GlobalPoint xyzzy((*simVertices)[ivert].position().x(),
                        (*simVertices)[ivert].position().y(),
                        (*simVertices)[ivert].position().z());
      initialPosition = xyzzy;
    }
    //
    //  Presumably t0 has dimensions of cm if not mm?
    //  Convert to ns for internal calculations.
    //  I wonder where we should get c from?
    //
    double tof = t0 / 29.98;

#ifdef FAMOS_DEBUG
    std::cout << " ===> MuonSimHitProducer::reconstruct() found SIMTRACK - pid = " << pid;
    std::cout << " : pT = " << mySimP4.Pt() << ", eta = " << mySimP4.Eta() << ", phi = " << mySimP4.Phi() << std::endl;
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
    tof += dtracker.mag() / 29.98;

#ifdef FAMOS_DEBUG
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
    Surface::RotationType rot = Surface::RotationType(xAxis, yAxis, zAxis);
    PlaneBuilder::ReturnType startingPlane = pb.plane(startingPosition, rot);
    GlobalTrajectoryParameters gtp(startingPosition, startingMomentum, (int)mySimTrack.charge(), magfield);
    TrajectoryStateOnSurface startingState(gtp, *startingPlane);

    std::vector<const DetLayer*> navLayers;
    if (fabs(startingState.globalMomentum().eta()) > 4.5) {
      navLayers = navigation.compatibleEndcapLayers(*(startingState.freeState()), alongMomentum);
    } else {
      navLayers = navigation.compatibleLayers(*(startingState.freeState()), alongMomentum);
    }
    /*
    edm::ESHandle<Propagator> propagator =
      theService->propagator("SteppingHelixPropagatorAny");
    */

    if (navLayers.empty())
      continue;

#ifdef FAMOS_DEBUG
    std::cout << "Found " << navLayers.size() << " compatible DetLayers..." << std::endl;
#endif

    TrajectoryStateOnSurface propagatedState = startingState;
    for (unsigned int ilayer = 0; ilayer < navLayers.size(); ilayer++) {
#ifdef FAMOS_DEBUG
      std::cout << "Propagating to layer " << ilayer << " " << dumper.dumpLayer(navLayers[ilayer]) << std::endl;
#endif

      std::vector<DetWithState> comps =
          navLayers[ilayer]->compatibleDets(propagatedState, *propagatorWithMaterial, theEstimator);
      if (comps.empty())
        continue;

#ifdef FAMOS_DEBUG
      std::cout << "Propagating " << propagatedState << std::endl;
#endif

      // Starting momentum
      double pi = propagatedState.globalMomentum().mag();

      // Propagate with material effects (dE/dx average only)
      SteppingHelixStateInfo shsStart(*(propagatedState.freeTrajectoryState()));
      SteppingHelixStateInfo shsDest;
      ((const SteppingHelixPropagator*)propagatorWithMaterial)
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
          propagatorWithoutMaterial->propagateWithPath(propagatedState, navLayers[ilayer]->surface());

      // Update the propagated state
      propagatedState = next.first;
      double pf = propagatedState.globalMomentum().mag();

      // Insert dE/dx fluctuations and multiple scattering
      // Skip this step if nextNoMaterial.first is not valid
      // This happens rarely (~0.02% of ttbar events)
      if (theMaterialEffects && nextNoMaterial.first.isValid())
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
      double m = mySimP4.M();
      double rbeta = sqrt(1 + m * m / (pavg * pavg)) / 29.98;
      double dtof = pathLength * rbeta;

#ifdef FAMOS_DEBUG
      std::cout << "Propagated to next surface... path length = " << pathLength << " cm, dTOF = " << dtof << " ns"
                << std::endl;
#endif

      tof += dtof;

      for (unsigned int icomp = 0; icomp < comps.size(); icomp++) {
        const GeomDet* gd = comps[icomp].first;
        if (gd->subDetector() == GeomDetEnumerators::DT) {
          DTChamberId id(gd->geographicalId());
          const DTChamber* chamber = dtGeom->chamber(id);
          std::vector<const DTSuperLayer*> superlayer = chamber->superLayers();
          for (unsigned int isl = 0; isl < superlayer.size(); isl++) {
            std::vector<const DTLayer*> layer = superlayer[isl]->layers();
            for (unsigned int ilayer = 0; ilayer < layer.size(); ilayer++) {
              DTLayerId lid = layer[ilayer]->id();
#ifdef FAMOS_DEBUG
              std::cout << "    Extrapolated to DT (" << lid.wheel() << "," << lid.station() << "," << lid.sector()
                        << "," << lid.superlayer() << "," << lid.layer() << ")" << std::endl;
#endif

              const GeomDetUnit* det = dtGeom->idToDetUnit(lid);

              HelixArbitraryPlaneCrossing crossing(propagatedState.globalPosition().basicVector(),
                                                   propagatedState.globalMomentum().basicVector(),
                                                   propagatedState.transverseCurvature(),
                                                   anyDirection);
              std::pair<bool, double> path = crossing.pathLength(det->surface());
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
              double theDTHitIneff = pmu > 0 ? exp(kDT * log(pmu) + fDT) : 0.;
              if (random.flatShoot() < theDTHitIneff)
                continue;

              double eloss = 0;
              double pz = fabs(lmom.z());
              LocalPoint entry = lpos - 0.5 * thickness * lmom / pz;
              LocalPoint exit = lpos + 0.5 * thickness * lmom / pz;
              double dtof = path.second * rbeta;
              int trkid = mySimTrack.trackId();
              unsigned int id = wid.rawId();
              short unsigned int processType = 2;
              PSimHit hit(
                  entry, exit, lmom.mag(), tof + dtof, eloss, pid, id, trkid, lmom.theta(), lmom.phi(), processType);
              theDTHits.push_back(hit);
            }
          }
        } else if (gd->subDetector() == GeomDetEnumerators::CSC) {
          CSCDetId id(gd->geographicalId());
          const CSCChamber* chamber = cscGeom->chamber(id);
          std::vector<const CSCLayer*> layers = chamber->layers();
          for (unsigned int ilayer = 0; ilayer < layers.size(); ilayer++) {
            CSCDetId lid = layers[ilayer]->id();

#ifdef FAMOS_DEBUG
            std::cout << "    Extrapolated to CSC (" << lid.endcap() << "," << lid.ring() << "," << lid.station() << ","
                      << lid.layer() << ")" << std::endl;
#endif

            const GeomDetUnit* det = cscGeom->idToDetUnit(lid);
            HelixArbitraryPlaneCrossing crossing(propagatedState.globalPosition().basicVector(),
                                                 propagatedState.globalMomentum().basicVector(),
                                                 propagatedState.transverseCurvature(),
                                                 anyDirection);
            std::pair<bool, double> path = crossing.pathLength(det->surface());
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
            double theCSCHitIneff = pmu > 0 ? exp(kCSC * log(pmu) + fCSC) : 0.;
            // Take into account the different geometry in ME11:
            if (id.station() == 1 && id.ring() == 1)
              theCSCHitIneff = theCSCHitIneff * 0.442;
            if (random.flatShoot() < theCSCHitIneff)
              continue;

            double eloss = 0;
            double pz = fabs(lmom.z());
            LocalPoint entry = lpos - 0.5 * thickness * lmom / pz;
            LocalPoint exit = lpos + 0.5 * thickness * lmom / pz;
            double dtof = path.second * rbeta;
            int trkid = mySimTrack.trackId();
            unsigned int id = lid.rawId();
            short unsigned int processType = 2;
            PSimHit hit(
                entry, exit, lmom.mag(), tof + dtof, eloss, pid, id, trkid, lmom.theta(), lmom.phi(), processType);
            theCSCHits.push_back(hit);
          }
        } else if (gd->subDetector() == GeomDetEnumerators::RPCBarrel ||
                   gd->subDetector() == GeomDetEnumerators::RPCEndcap) {
          RPCDetId id(gd->geographicalId());
          const RPCChamber* chamber = rpcGeom->chamber(id);
          std::vector<const RPCRoll*> roll = chamber->rolls();
          for (unsigned int iroll = 0; iroll < roll.size(); iroll++) {
            RPCDetId rid = roll[iroll]->id();

#ifdef FAMOS_DEBUG
            std::cout << "    Extrapolated to RPC (" << rid.ring() << "," << rid.station() << "," << rid.sector() << ","
                      << rid.subsector() << "," << rid.layer() << "," << rid.roll() << ")" << std::endl;
#endif

            const GeomDetUnit* det = rpcGeom->idToDetUnit(rid);
            HelixArbitraryPlaneCrossing crossing(propagatedState.globalPosition().basicVector(),
                                                 propagatedState.globalMomentum().basicVector(),
                                                 propagatedState.transverseCurvature(),
                                                 anyDirection);
            std::pair<bool, double> path = crossing.pathLength(det->surface());
            if (!path.first)
              continue;
            LocalPoint lpos = det->toLocal(GlobalPoint(crossing.position(path.second)));
            if (!det->surface().bounds().inside(lpos))
              continue;
            double thickness = det->surface().bounds().thickness();
            LocalVector lmom = det->toLocal(GlobalVector(crossing.direction(path.second)));
            lmom = lmom.unit() * propagatedState.localMomentum().mag();
            double eloss = 0;
            double pz = fabs(lmom.z());
            LocalPoint entry = lpos - 0.5 * thickness * lmom / pz;
            LocalPoint exit = lpos + 0.5 * thickness * lmom / pz;
            double dtof = path.second * rbeta;
            int trkid = mySimTrack.trackId();
            unsigned int id = rid.rawId();
            short unsigned int processType = 2;
            PSimHit hit(
                entry, exit, lmom.mag(), tof + dtof, eloss, pid, id, trkid, lmom.theta(), lmom.phi(), processType);
            theRPCHits.push_back(hit);
          }
        } else {
          std::cout << "Extrapolated to unknown subdetector '" << gd->subDetector() << "'..." << std::endl;
        }
      }
    }
  }

  std::unique_ptr<edm::PSimHitContainer> pcsc(new edm::PSimHitContainer);
  int n = 0;
  for (std::vector<PSimHit>::const_iterator i = theCSCHits.begin(); i != theCSCHits.end(); i++) {
    pcsc->push_back(*i);
    n += 1;
  }
  iEvent.put(std::move(pcsc), "MuonCSCHits");

  std::unique_ptr<edm::PSimHitContainer> pdt(new edm::PSimHitContainer);
  n = 0;
  for (std::vector<PSimHit>::const_iterator i = theDTHits.begin(); i != theDTHits.end(); i++) {
    pdt->push_back(*i);
    n += 1;
  }
  iEvent.put(std::move(pdt), "MuonDTHits");

  std::unique_ptr<edm::PSimHitContainer> prpc(new edm::PSimHitContainer);
  n = 0;
  for (std::vector<PSimHit>::const_iterator i = theRPCHits.begin(); i != theRPCHits.end(); i++) {
    prpc->push_back(*i);
    n += 1;
  }
  iEvent.put(std::move(prpc), "MuonRPCHits");
}

void MuonSimHitProducer::readParameters(const edm::ParameterSet& fastMuons,
                                        const edm::ParameterSet& fastTracks,
                                        const edm::ParameterSet& matEff) {
  // Muons
  std::string _simModuleLabel = fastMuons.getParameter<std::string>("simModuleLabel");
  std::string _simModuleProcess = fastMuons.getParameter<std::string>("simModuleProcess");
  simMuonLabel = edm::InputTag(_simModuleLabel, _simModuleProcess);
  simVertexLabel = edm::InputTag(_simModuleLabel);

  std::vector<double> simHitIneffDT = fastMuons.getParameter<std::vector<double> >("simHitDTIneffParameters");
  std::vector<double> simHitIneffCSC = fastMuons.getParameter<std::vector<double> >("simHitCSCIneffParameters");
  kDT = simHitIneffDT[0];
  fDT = simHitIneffDT[1];
  kCSC = simHitIneffCSC[0];
  fCSC = simHitIneffCSC[1];

  // Tracks
  fullPattern_ = fastTracks.getUntrackedParameter<bool>("FullPatternRecognition");

  // The following should be on LogInfo
  //  std::cout << " MUON SIM HITS: FastSimulation parameters " << std::endl;
  //  std::cout << " ============================================== " << std::endl;
  //  if ( fullPattern_ )
  //    std::cout << " The FULL pattern recognition option is turned ON" << std::endl;
  //  else
  //    std::cout << " The FAST tracking option is turned ON" << std::endl;

  // Material Effects
  theMaterialEffects = nullptr;
  if (matEff.getParameter<bool>("PairProduction") || matEff.getParameter<bool>("Bremsstrahlung") ||
      matEff.getParameter<bool>("MuonBremsstrahlung") || matEff.getParameter<bool>("EnergyLoss") ||
      matEff.getParameter<bool>("MultipleScattering"))
    theMaterialEffects = new MaterialEffects(matEff);
}

void MuonSimHitProducer::applyMaterialEffects(TrajectoryStateOnSurface& tsosWithdEdx,
                                              TrajectoryStateOnSurface& tsos,
                                              double radPath,
                                              RandomEngineAndDistribution const* random,
                                              HepPDT::ParticleDataTable const& table) {
  // The energy loss simulator
  EnergyLossSimulator* energyLoss = theMaterialEffects->energyLossSimulator();

  // The multiple scattering simulator
  MultipleScatteringSimulator* multipleScattering = theMaterialEffects->multipleScatteringSimulator();

  // The Muon Bremsstrahlung simulator
  MuonBremsstrahlungSimulator* bremsstrahlung = theMaterialEffects->muonBremsstrahlungSimulator();

  // Initialize the Particle position, momentum and energy
  const Surface& nextSurface = tsos.surface();
  GlobalPoint gPos = energyLoss ? tsos.globalPosition() : tsosWithdEdx.globalPosition();
  GlobalVector gMom = energyLoss ? tsos.globalMomentum() : tsosWithdEdx.globalMomentum();
  double mu = 0.1056583692;
  double en = std::sqrt(gMom.mag2() + mu * mu);

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
    double enWithdEdx = std::sqrt(gMomWithdEdx.mag2() + mu * mu);
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
    fac = (theNewMomentum.E() * theNewMomentum.E() - mu * mu) / theNewMomentum.Vect().Mag2();
    fac = fac > 0. ? std::sqrt(fac) : 1E-9;
    theMuon.particle().setMomentum(
        theNewMomentum.Px() * fac, theNewMomentum.Py() * fac, theNewMomentum.Pz() * fac, theNewMomentum.E());
    theMuon.particle().setVertex(theNewPosition);
  }

  // Does the actual mutliple scattering
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
  GlobalTrajectoryParameters propagatedGtp(propagatedPosition, propagatedMomentum, (int)charge, magfield);
  tsosWithdEdx = TrajectoryStateOnSurface(propagatedGtp, nextSurface);
}

//define this as a plug-in
DEFINE_FWK_MODULE(MuonSimHitProducer);
