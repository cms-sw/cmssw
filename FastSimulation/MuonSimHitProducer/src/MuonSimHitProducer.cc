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
// $Id: MuonSimHitProducer.cc,v 1.6 2007/11/10 20:46:14 mulders Exp $
//
//

// CMSSW headers 
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/PluginManager/interface/PluginManager.h"

// Fast Simulation headers
#include "FastSimulation/Utilities/interface/RandomEngine.h"
#include "FastSimulation/MuonSimHitProducer/interface/MuonSimHitProducer.h"

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

// Tracking Tools
#include "TrackingTools/GeomPropagators/interface/HelixArbitraryPlaneCrossing.h"

// Data Formats
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/GeometrySurface/interface/PlaneBuilder.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "RecoMuon/TrackingTools/interface/MuonTrajectoryUpdator.h"
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"

////////////////////////////////////////////////////////////////////////////
// Geometry, Magnetic Field
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

// Root
#include <TRandom3.h>

////////////////////// Now find detector IDs:

// #include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"


//
// constructors and destructor
//
MuonSimHitProducer::MuonSimHitProducer(const edm::ParameterSet& iConfig) {

  readParameters(iConfig.getParameter<edm::ParameterSet>("MUONS"),
		 iConfig.getParameter<edm::ParameterSet>("TRACKS"));

//
//  register your products ... need to declare at least one possible product...
//
  produces<edm::PSimHitContainer>("MuonCSCHits");
  produces<edm::PSimHitContainer>("MuonDTHits");
  produces<edm::PSimHitContainer>("MuonRPCHits");

  edm::ParameterSet serviceParameters =
     iConfig.getParameter<edm::ParameterSet>("ServiceParameters");
  theService = new MuonServiceProxy(serviceParameters);
  edm::ParameterSet updatorParameters = 
     iConfig.getParameter<edm::ParameterSet>("MuonTrajectoryUpdatorParameters");
  theUpdator = new MuonTrajectoryUpdator(updatorParameters,insideOut);
//
//  Initialize the random number generator service
//

  edm::Service<edm::RandomNumberGenerator> rng;
  if ( ! rng.isAvailable() ) {
    throw cms::Exception("Configuration") <<
      "MuonSimHitProducer requires the RandomGeneratorService \n"
      "which is not present in the configuration file. \n"
      "You must add the service in the configuration file\n"
      "or remove the module that requires it.";
  }

  bool useTRandom = iConfig.getParameter<bool>("UseTRandomEngine");
  if ( !useTRandom ) { 
    random = new RandomEngine(&(*rng));
  } else {
    TRandom3* anEngine = new TRandom3();
    anEngine->SetSeed(rng->mySeed());
    random = new RandomEngine(anEngine);
  }

  random = new RandomEngine(&(*rng));
}

// ---- method called once each job just before starting event loop ----
void 
MuonSimHitProducer::beginJob (edm::EventSetup const & es) {

  //services

  edm::ESHandle<MagneticField>          magField;
  edm::ESHandle<DTGeometry>             dtGeometry;
  edm::ESHandle<CSCGeometry>            cscGeometry;
  edm::ESHandle<RPCGeometry>            rpcGeometry;

  es.get<IdealMagneticFieldRecord>().get(magField);
  es.get<MuonGeometryRecord>().get(dtGeometry);
  es.get<MuonGeometryRecord>().get(cscGeometry);
  es.get<MuonGeometryRecord>().get(rpcGeometry);

  magfield = &(*magField);
  dtGeom = &(*dtGeometry);
  cscGeom = &(*cscGeometry);
  rpcGeom = &(*rpcGeometry);

  theService->update(es);

}
  
MuonSimHitProducer::~MuonSimHitProducer()
{
  // do anything here that needs to be done at destruction time
  // (e.g. close files, deallocate resources etc.)
  
  if ( random ) { 
    if ( random->theRootEngine() ) delete random->theRootEngine();
    delete random;
  }
}


//
// member functions
//

// ------------ method called to produce the data  ------------

void MuonSimHitProducer::produce(edm::Event& iEvent,
                                 const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace std;

  MuonPatternRecoDumper dumper;

  Handle<vector<SimTrack> > simMuons;
  Handle<vector<SimVertex> > simVertices;
  vector<PSimHit> theCSCHits;
  vector<PSimHit> theDTHits;
  vector<PSimHit> theRPCHits;

  DirectMuonNavigation navigation(theService->detLayerGeometry());
  iEvent.getByLabel(theSimModuleLabel_,theSimModuleProcess_,simMuons);
  iEvent.getByLabel(theSimModuleLabel_,simVertices);

  for ( unsigned int itrk=0; itrk<simMuons->size(); itrk++ ) {
    const SimTrack &mySimTrack = (*simMuons)[itrk];
    math::XYZTLorentzVector mySimP4(mySimTrack.momentum().x(),
                                    mySimTrack.momentum().y(),
                                    mySimTrack.momentum().z(),
                                    mySimTrack.momentum().t());
    double t0 = 0;
    GlobalPoint initialPosition;
    int ivert = mySimTrack.vertIndex();
    if ( ivert >= 0 ) {
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
    double tof = t0/29.98;

    if ( debug_ ) {
      cout << " ===> MuonSimHitProducer::reconstruct() found SIMTRACK - pid = "
           << mySimTrack.type() ;
      cout << " : pT = " << mySimP4.Pt()
           << ", eta = " << mySimP4.Eta()
           << ", phi = " << mySimP4.Phi() << endl;
    }

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
    GlobalVector dtracker = startingPosition-initialPosition;
    tof += dtracker.mag()/29.98;

    if ( debug_ ) {
      cout << " the Muon START position " << startingPosition << endl;
      cout << " the Muon START momentum " << startingMomentum << endl;
    }

// 
//  Some magic to define a TrajectoryStateOnSurface
//
    PlaneBuilder pb;
    GlobalVector zAxis = startingMomentum.unit();
    GlobalVector yAxis(zAxis.y(),-zAxis.x(),0); 
    GlobalVector xAxis = yAxis.cross(zAxis);
    Surface::RotationType rot = Surface::RotationType(xAxis,yAxis,zAxis);
    PlaneBuilder::ReturnType startingPlane = pb.plane(startingPosition,rot);
    GlobalTrajectoryParameters gtp(startingPosition,
                                   startingMomentum,
                                   (int)mySimTrack.charge(),
                                   magfield);
    TrajectoryStateOnSurface startingState(gtp,*startingPlane);

    vector<const DetLayer *> navLayers;
    if ( fabs(startingState.globalMomentum().eta()) > 4.5 ) {
      navLayers = navigation.compatibleEndcapLayers(*(startingState.freeState()),
                                                    alongMomentum);
    }
    else {
      navLayers = navigation.compatibleLayers(*(startingState.freeState()),
                                               alongMomentum);
    }
    ESHandle<Propagator> propagator =
      theService->propagator("SteppingHelixPropagatorAny");

    if ( navLayers.empty() ) continue;

    if ( debug_ ) {
      cout << "Found " << navLayers.size()
           << " compatible DetLayers..." << endl;
    }
    TrajectoryStateOnSurface propagatedState = startingState;
    for ( unsigned int ilayer=0; ilayer<navLayers.size(); ilayer++ ) {
      if ( debug_ ) {
        cout << "Propagating to layer " << ilayer << " " << dumper.dumpLayer(navLayers[ilayer]) << endl;
      }
      vector<DetWithState> comps = navLayers[ilayer]->compatibleDets(propagatedState,*propagator,*(theUpdator->estimator()));
      if ( comps.empty() ) continue;
      if ( debug_ ) {
        cout << "Propagating " << propagatedState << endl;
      }
      pair<TrajectoryStateOnSurface,double> next = propagator->propagateWithPath(propagatedState,navLayers[ilayer]->surface());
      double pi = propagatedState.globalMomentum().mag();
      if ( next.first.isValid() ) {
        propagatedState = next.first;
      }
//
//  Consider this... 1 GeV muon has a velocity that is only 0.5% slower than c...
//  We probably can safely ignore the mass for anything that makes it out to the
//  muon chambers.
//
      double pf = propagatedState.globalMomentum().mag();
      double pavg = 0.5*(pi+pf);
      double m = mySimP4.M();
      double rbeta = sqrt(1+m*m/(pavg*pavg))/29.98;
      double dtof = next.second*rbeta;
      if ( debug_ ) {
        cout << "Propagated to next surface... path length = " << next.second << " cm, dTOF = " << dtof << " ns" << endl;
      }
      tof += dtof;
      const GeomDet *gd = comps[0].first;
      if ( gd->subDetector() == GeomDetEnumerators::DT ) {
        DTChamberId id(gd->geographicalId());
        const DTChamber *chamber = dtGeom->chamber(id);
        vector<const DTSuperLayer *> superlayer = chamber->superLayers();
        for ( unsigned int isl=0; isl<superlayer.size(); isl++ ) {
          vector<const DTLayer *> layer = superlayer[isl]->layers();
          for ( unsigned int ilayer=0; ilayer<layer.size(); ilayer++ ) {
            DTLayerId lid = layer[ilayer]->id();
            if ( debug_ ) {
              cout << "    Extrapolated to DT (" << lid.wheel() << "," 
                                                 << lid.station() << "," 
                                                 << lid.sector() << "," 
                                                 << lid.superlayer() << "," 
                                                 << lid.layer() << ")" << endl;
            }
            const GeomDetUnit *det = dtGeom->idToDetUnit(lid);

            HelixArbitraryPlaneCrossing crossing(propagatedState.globalPosition().basicVector(),
                                                 propagatedState.globalMomentum().basicVector(),
                                                 propagatedState.transverseCurvature(),
                                                 anyDirection);
            pair<bool,double> path = crossing.pathLength(det->surface());
            if ( ! path.first ) continue;
            LocalPoint lpos = det->toLocal(GlobalPoint(crossing.position(path.second)));
            if ( fabs(lpos.x()) > 0.5*det->surface().bounds().width() ||
                 fabs(lpos.y()) > 0.5*det->surface().bounds().length() ) continue;
//
//  The use of the channel() method claims to be deprecated in DTTopology...
//
            const DTTopology& dtTopo = layer[ilayer]->specificTopology();
            int wire = dtTopo.channel(lpos);
	    if (wire < dtTopo.firstChannel()) {
	      cout << "DT wire number too low; check DTTopology.channel() method !!" << endl;
	      wire = dtTopo.firstChannel();
	    }
	    if (wire > dtTopo.lastChannel()) {
	      cout << "DT wire number too high; check DTTopology.channel() method !!" << endl;
	      wire = dtTopo.lastChannel();	      
	    }
//
//  The wire number calculation is somewhat imperical at this point...  The
//  drift cell width is 4.22 cm, but the absolute offset needs to be checked.
//
//          int wire = round((lpos.x()+0.5*det->surface().bounds().width())/4.22);
//
            DTWireId wid(lid,wire);
            double thickness = det->surface().bounds().thickness();
            LocalVector lmom = det->toLocal(GlobalVector(crossing.direction(path.second)));
            lmom = lmom.unit()*propagatedState.localMomentum().mag();
            double eloss = 0;
            double pz = fabs(lmom.z());
            LocalPoint entry = lpos - 0.5*thickness*lmom/pz;
            LocalPoint exit = lpos + 0.5*thickness*lmom/pz;
            double dtof = path.second*rbeta;
            int pid = mySimTrack.type();
            int trkid = itrk;
            unsigned int id = wid.rawId();
	    short unsigned int processType = 2;
            PSimHit hit(entry,exit,lmom.mag(),
                        tof+dtof,eloss,pid,id,trkid,lmom.theta(),lmom.phi(),processType);
            theDTHits.push_back(hit);

          }
        }
      }
      else if ( gd->subDetector() == GeomDetEnumerators::CSC ) {
        CSCDetId id(gd->geographicalId());
        const CSCChamber *chamber = cscGeom->chamber(id);
        vector<const CSCLayer *> layer = chamber->layers();
        for ( unsigned int ilayer=0; ilayer<layer.size(); ilayer++ ) {
          CSCDetId lid = layer[ilayer]->id();
          if ( debug_ ) {
            cout << "    Extrapolated to CSC (" << lid.endcap() << "," 
                                                << lid.ring() << "," 
                                                << lid.station() << "," 
                                                << lid.layer() << ")" << endl;
          }
          const GeomDetUnit *det = cscGeom->idToDetUnit(lid);
          HelixArbitraryPlaneCrossing crossing(propagatedState.globalPosition().basicVector(),
                                               propagatedState.globalMomentum().basicVector(),
                                               propagatedState.transverseCurvature(),
                                               anyDirection);
          pair<bool,double> path = crossing.pathLength(det->surface());
          if ( ! path.first ) continue;
          LocalPoint lpos = det->toLocal(GlobalPoint(crossing.position(path.second)));
          if ( fabs(lpos.x()) > 0.5*det->surface().bounds().width() ||
               fabs(lpos.y()) > 0.5*det->surface().bounds().length() ) continue;
          double thickness = det->surface().bounds().thickness();
          LocalVector lmom = det->toLocal(GlobalVector(crossing.direction(path.second)));
          lmom = lmom.unit()*propagatedState.localMomentum().mag();
          double eloss = 0;
          double pz = fabs(lmom.z());
          LocalPoint entry = lpos - 0.5*thickness*lmom/pz;
          LocalPoint exit = lpos + 0.5*thickness*lmom/pz;
          double dtof = path.second*rbeta;
          int pid = mySimTrack.type();
          int trkid = itrk;
          unsigned int id = lid.rawId();
	  short unsigned int processType = 2;
          PSimHit hit(entry,exit,lmom.mag(),
                      tof+dtof,eloss,pid,id,trkid,lmom.theta(),lmom.phi(),processType);
          theCSCHits.push_back(hit);
        }
      }
      else if ( gd->subDetector() == GeomDetEnumerators::RPCBarrel ||
                gd->subDetector() == GeomDetEnumerators::RPCEndcap ) {
        RPCDetId id(gd->geographicalId());
        const RPCChamber *chamber = rpcGeom->chamber(id);
        vector<const RPCRoll *> roll = chamber->rolls();
        for ( unsigned int iroll=0; iroll<roll.size(); iroll++ ) {
          RPCDetId rid = roll[iroll]->id();
          if ( debug_ ) {
            cout << "    Extrapolated to RPC (" << rid.ring() << "," 
                                                << rid.station() << ","
                                                << rid.sector() << ","
                                                << rid.subsector() << ","
                                                << rid.layer() << ","
                                                << rid.roll() << ")" << endl;
          }
          const GeomDetUnit *det = rpcGeom->idToDetUnit(rid);
          HelixArbitraryPlaneCrossing crossing(propagatedState.globalPosition().basicVector(),
                                               propagatedState.globalMomentum().basicVector(),
                                               propagatedState.transverseCurvature(),
                                               anyDirection);
          pair<bool,double> path = crossing.pathLength(det->surface());
          if ( ! path.first ) continue;
          LocalPoint lpos = det->toLocal(GlobalPoint(crossing.position(path.second)));
          if ( fabs(lpos.x()) > 0.5*det->surface().bounds().width() ||
               fabs(lpos.y()) > 0.5*det->surface().bounds().length() ) continue;
          double thickness = det->surface().bounds().thickness();
          LocalVector lmom = det->toLocal(GlobalVector(crossing.direction(path.second)));
          lmom = lmom.unit()*propagatedState.localMomentum().mag();
          double eloss = 0;
          double pz = fabs(lmom.z());
          LocalPoint entry = lpos - 0.5*thickness*lmom/pz;
          LocalPoint exit = lpos + 0.5*thickness*lmom/pz;
          double dtof = path.second*rbeta;
          int pid = mySimTrack.type();
          int trkid = itrk;
          unsigned int id = rid.rawId();
	  short unsigned int processType = 2;
          PSimHit hit(entry,exit,lmom.mag(),
                      tof+dtof,eloss,pid,id,trkid,lmom.theta(),lmom.phi(),processType);
          theRPCHits.push_back(hit);
        }
      }
      else {
        cout << "Extrapolated to unknown subdetector '" << gd->subDetector() << "'..." << endl;
      }
    }
  }

  std::auto_ptr<edm::PSimHitContainer> pcsc(new edm::PSimHitContainer);
  int n = 0;
  for ( vector<PSimHit>::const_iterator i = theCSCHits.begin();
        i != theCSCHits.end(); i++ ) {
    pcsc->push_back(*i);
    n += 1;
  }
  iEvent.put(pcsc,"MuonCSCHits");

  std::auto_ptr<edm::PSimHitContainer> pdt(new edm::PSimHitContainer);
  n = 0;
  for ( vector<PSimHit>::const_iterator i = theDTHits.begin();
        i != theDTHits.end(); i++ ) {
    pdt->push_back(*i);
    n += 1;
  }
  iEvent.put(pdt,"MuonDTHits");

  std::auto_ptr<edm::PSimHitContainer> prpc(new edm::PSimHitContainer);
  n = 0;
  for ( vector<PSimHit>::const_iterator i = theRPCHits.begin();
        i != theRPCHits.end(); i++ ) {
    prpc->push_back(*i);
    n += 1;
  }
  iEvent.put(prpc,"MuonRPCHits");

}


// ------------ method called once each job just after ending the event loop  ------------
void MuonSimHitProducer::endJob() 
{
}


void MuonSimHitProducer::readParameters(const edm::ParameterSet& fastMuons, 
                                        const edm::ParameterSet& fastTracks) {
  // Muons
  debug_ = fastMuons.getUntrackedParameter<bool>("Debug");
  theSimModuleLabel_ = fastMuons.getParameter<std::string>("simModuleLabel");
  theSimModuleProcess_ = fastMuons.getParameter<std::string>("simModuleProcess");
  theTrkModuleLabel_ = fastMuons.getParameter<std::string>("trackModuleLabel");
  minEta_ = fastMuons.getParameter<double>("MinEta");
  maxEta_ = fastMuons.getParameter<double>("MaxEta");
  if (minEta_ > maxEta_) {
    double tempEta_ = maxEta_ ;
    maxEta_ = minEta_ ;
    minEta_ = tempEta_ ;
  }

  // Tracks
  fullPattern_  = fastTracks.getUntrackedParameter<bool>("FullPatternRecognition");

  std::cout << " MUON SIM HITS: FastSimulation parameters " << std::endl;
  std::cout << " ============================================== " << std::endl;
  std::cout << " Sim Hits produced for muons in the pseudorapidity range : "
            << minEta_ << " -> " << maxEta_ << std::endl;
  if ( fullPattern_ ) 
    std::cout << " The FULL pattern recognition option is turned ON" << std::endl;
  else
    std::cout << " The FAST tracking option is turned ON" << std::endl;
}


//define this as a plug-in
DEFINE_FWK_MODULE(MuonSimHitProducer);
