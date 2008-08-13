// -*- C++ -*-
//
// Package:    MuonHIPOverlapsStripwiseRefitter
// Class:      MuonHIPOverlapsStripwiseRefitter
// 
/**\class MuonHIPOverlapsStripwiseRefitter MuonHIPOverlapsStripwiseRefitter.cc Alignment/MuonHIPOverlapsStripwiseRefitter/src/MuonHIPOverlapsStripwiseRefitter.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jim Pivarski
//         Created:  Wed Dec 12 13:31:55 CST 2007
// $Id: MuonHIPOverlapsStripwiseRefitter.cc,v 1.1 2008/08/13 00:06:14 pivarski Exp $
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// references
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHitBuilder.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "CondFormats/Alignment/interface/Definitions.h"
#include "DataFormats/GeometrySurface/interface/Surface.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"
#include "TTree.h"

// products
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

//
// class decleration
//

class MuonHIPOverlapsStripwiseRefitter : public edm::EDProducer {
   public:
      explicit MuonHIPOverlapsStripwiseRefitter(const edm::ParameterSet&);
      ~MuonHIPOverlapsStripwiseRefitter();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      // ----------member data ---------------------------
      edm::InputTag m_input;
      unsigned int m_minHits;

      bool m_debuggingPrintouts, m_debuggingNtuple;
      TTree *m_ntuple;
      Int_t m_ntuple_endcap, m_ntuple_station, m_ntuple_ring, m_ntuple_chamber, m_ntuple_layer, m_ntuple_strip;
      Float_t m_ntuple_trackx, m_ntuple_tracky, m_ntuple_hitx, m_ntuple_hity, m_ntuple_stripAngle;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
MuonHIPOverlapsStripwiseRefitter::MuonHIPOverlapsStripwiseRefitter(const edm::ParameterSet& iConfig)
   : m_input(iConfig.getParameter<edm::InputTag>("input"))
   , m_minHits(iConfig.getParameter<unsigned int>("minHits"))
   , m_debuggingPrintouts(iConfig.getUntrackedParameter<bool>("debuggingPrintouts", false))
   , m_debuggingNtuple(iConfig.getUntrackedParameter<bool>("debuggingNtuple", false))
{
   if (m_debuggingNtuple) {
      edm::Service<TFileService> tfile;
      m_ntuple = tfile->make<TTree>("debuggingNtuple", "debuggingNtuple");
      m_ntuple->Branch("endcap", &m_ntuple_endcap, "endcap/I");
      m_ntuple->Branch("station", &m_ntuple_station, "station/I");
      m_ntuple->Branch("ring", &m_ntuple_ring, "ring/I");
      m_ntuple->Branch("chamber", &m_ntuple_chamber, "chamber/I");
      m_ntuple->Branch("layer", &m_ntuple_layer, "layer/I");
      m_ntuple->Branch("strip", &m_ntuple_strip, "strip/I");
      m_ntuple->Branch("trackx", &m_ntuple_trackx, "trackx/F");
      m_ntuple->Branch("tracky", &m_ntuple_tracky, "tracky/F");
      m_ntuple->Branch("hitx", &m_ntuple_hitx, "hitx/F");
      m_ntuple->Branch("hity", &m_ntuple_hity, "hity/F");
      m_ntuple->Branch("stripAngle", &m_ntuple_stripAngle, "stripAngle/F");
   }

   produces<std::vector<Trajectory> >();
   produces<TrajTrackAssociationCollection>();
}


MuonHIPOverlapsStripwiseRefitter::~MuonHIPOverlapsStripwiseRefitter()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
MuonHIPOverlapsStripwiseRefitter::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   if (m_debuggingPrintouts) std::cout << "Begin working with an event" << std::endl;

   edm::Handle<reco::TrackCollection> tracks;
   iEvent.getByLabel(m_input, tracks);

   edm::ESHandle<CSCGeometry> cscGeometry;
   iSetup.get<MuonGeometryRecord>().get(cscGeometry);

   // only a formal requirement; not used
   edm::ESHandle<MagneticField> magneticField;
   iSetup.get<IdealMagneticFieldRecord>().get(magneticField);

   edm::ESHandle<GlobalTrackingGeometry> globalGeometry;
   iSetup.get<GlobalTrackingGeometryRecord>().get(globalGeometry);

   // Create these factories once per event
   TrajectoryStateTransform transformer;
   MuonTransientTrackingRecHitBuilder muonTransBuilder;

   // Create a collection of Trajectories, to put in the Event
   std::auto_ptr<std::vector<Trajectory> > trajectoryCollection(new std::vector<Trajectory>);

   // Remember which trajectory is associated with which track
   std::map<edm::Ref<std::vector<Trajectory> >::key_type, edm::Ref<reco::TrackCollection>::key_type> reference_map;
   edm::Ref<std::vector<Trajectory> >::key_type trajCounter = 0;
   edm::Ref<reco::TrackCollection>::key_type trackCounter = 0;

   for (reco::TrackCollection::const_iterator track = tracks->begin();  track != tracks->end();  ++track) {
      if (m_debuggingPrintouts) std::cout << "Begin working with a track" << std::endl;

      trackCounter++;
      
      std::vector<std::vector<const TrackingRecHit*> > hits_by_station;
      std::vector<const TrackingRecHit*> current_station;
      int last_station = 0;
      DetId last_id;

      for (trackingRecHit_iterator hit = track->recHitsBegin();  hit != track->recHitsEnd();  ++hit) {
	 DetId id = (*hit)->geographicalId();
	 if (id.det() == DetId::Muon  &&  id.subdetId() == MuonSubdetId::CSC) {

	    CSCDetId cscId(id.rawId());
	    int station = (cscId.endcap() == 1 ? 1 : -1) * cscId.station();

	    if (last_station == 0) last_station = station;
	    if (last_station != station) {

	       if (current_station.size() >= m_minHits) {
		  hits_by_station.push_back(current_station);
	       }
	       current_station.clear();
	    }

	    current_station.push_back(&**hit);
	    last_station = station;
	    last_id = id;
	 } // end if CSC
      } // end loop over hits

      // add the last station
      if (last_id.subdetId() == MuonSubdetId::CSC) {
	 if (current_station.size() >= m_minHits) {
	    hits_by_station.push_back(current_station);
	 }
      }

      // these must be in lock-step
      edm::OwnVector<TrackingRecHit> clonedHits;
      std::vector<TrajectoryMeasurement::ConstRecHitPointer> transHits;
      std::vector<TrajectoryStateOnSurface> TSOSes;

      for (std::vector<std::vector<const TrackingRecHit*> >::const_iterator station = hits_by_station.begin();  station != hits_by_station.end();  ++station) {
	 // two fits: one from even to odd and another from odd to even
	 for (int bit = 0;  bit <= 1;  ++bit) {

	    if (m_debuggingPrintouts) std::cout << "Fitting " << (bit == 0? "even": "odd") << ", making residuals on " << (bit == 0? "odd": "even") << std::endl;

	    double SXX, SxXX, SxXY, SXY, SxzXX, SxzXY, SyXY, SYY, SyYY, SyzXY, SyzYY, SzXX, SzXY, SzYY, SzzXX, SzzXY, SzzYY;
	    SXX = SxXX = SxXY = SXY = SxzXX = SxzXY = SyXY = SYY = SyYY = SyzXY = SyzYY = SzXX = SzXY = SzYY = SzzXX = SzzXY = SzzYY = 0.;

	    // the chamber we align is the master coordinate system given by chamberSurface
	    // the chamber in which we fit needs to be transformed to that system
	    const Surface* chamberSurface = NULL;
	    for (std::vector<const TrackingRecHit*>::const_iterator hit = station->begin();  hit != station->end();  ++hit) {
	       CSCDetId id((*hit)->geographicalId());
	       if (id.chamber() % 2 != bit) {  // if it doesn't equal bit, don't fit!
		  CSCDetId chamberId(id.endcap(), id.station(), id.ring(), id.chamber(), 0);
		  chamberSurface = &(cscGeometry->idToDet(chamberId)->surface());
		  
		  if (m_debuggingPrintouts) std::cout << "Mutual coordinate system is " << id << std::endl;

		  break;
	       }
	    }
	    assert(chamberSurface);
	    
	    // collecting hits for the FIT
	    for (std::vector<const TrackingRecHit*>::const_iterator hit = station->begin();  hit != station->end();  ++hit) {
	       CSCDetId id((*hit)->geographicalId());
	       if (id.chamber() % 2 == bit) {  // if it equals bit, fit!

		  if (m_debuggingPrintouts) std::cout << "    adding hit on " << id << " to the fit" << std::endl;

		  LocalPoint localPoint = (*hit)->localPosition();
	       
		  TrajectoryMeasurement::ConstRecHitPointer hitPtr(muonTransBuilder.build(&**hit, globalGeometry));
		  AlgebraicSymMatrix localErrorWithAPE = hitPtr->parametersError();
		  double sigma_xx = localErrorWithAPE[0][0];
		  double sigma_xy = (localErrorWithAPE.num_row() == 1 ? 0. : localErrorWithAPE[0][1]);
		  double sigma_yy = (localErrorWithAPE.num_row() == 1 ? 0. : localErrorWithAPE[1][1]);
		     
		  const Surface& layerSurface = cscGeometry->idToDet(id)->surface();
		  LocalPoint chamberPoint = chamberSurface->toLocal(layerSurface.toGlobal(localPoint));

		  AlgebraicSymMatrix chamberError(2);
		  align::RotationType rotation = chamberSurface->rotation() * layerSurface.rotation().transposed();
		  align::RotationType error = rotation * align::RotationType(sigma_xx, sigma_xy, 0, sigma_xy, sigma_yy, 0, 0, 0, 0) * rotation.transposed();
		  chamberError[0][0] = error.xx();
		  chamberError[1][1] = error.yy();
		  chamberError[0][1] = error.yx();

		  int ierr;
		  chamberError.invert(ierr);
		  if (ierr != 0) {
		     edm::LogError("MuonHIPOverlapsStripwiseRefitter") << "Matrix inversion failed!  ierr = " << ierr << " matrix = " << std::endl << chamberError << std::endl;
		     return;
		  }

		  double xi = chamberPoint.x();
		  double yi = chamberPoint.y();
		  double zi = chamberPoint.z();
		  double XX = chamberError[0][0];
		  double XY = chamberError[0][1];
		  double YY = chamberError[1][1];

		  SXX += XX;
		  SxXX += xi * XX;
		  SxXY += xi * XY;
		  SXY += XY;
		  SxzXX += xi * zi * XX;
		  SxzXY += xi * zi * XY;
		  SyXY += yi * XY;
		  SYY += YY;
		  SyYY += yi * YY;
		  SyzXY += yi * zi * XY;
		  SyzYY += yi * zi * YY;
		  SzXX += zi * XX;
		  SzXY += zi * XY;
		  SzYY += zi * YY;
		  SzzXX += zi * zi * XX;
		  SzzXY += zi * zi * XY;
		  SzzYY += zi * zi * YY;
	       } // end if this is a hit for fitting
	    } // end loop over hits

	    // calculate the least-squares fit
	    double denom = (SzzXX*(SXX*(SzzYY*SYY - pow(SzYY,2)) - pow(SzXY,2)*SYY - SzzYY*pow(SXY,2) + 2*SzXY*SzYY*SXY) + SzzXY*(SzXY*(2*SzXX*SYY + 2*SzYY*SXX) - 2*SzXX*SzYY*SXY - 2*pow(SzXY,2)*SXY) + pow(SzzXY,2)*(pow(SXY,2) - SXX*SYY) + pow(SzXX,2)*(pow(SzYY,2) - SzzYY*SYY) + 2*SzXX*SzXY*SzzYY*SXY + pow(SzXY,2)*(-SzzYY*SXX - 2*SzXX*SzYY) + pow(SzXY,4));
	    double a = (-SzXX*(SyXY*(SzzYY*SYY - pow(SzYY,2)) + SxXX*(SzzYY*SYY - pow(SzYY,2)) + ((-SyYY - SxXY)*SzzYY + (SyzYY + SxzXY)*SzYY)*SXY) - SyzXY*(SXX*(pow(SzYY,2) - SzzYY*SYY) + SzzYY*pow(SXY,2)) - SxzXX*(SXX*(pow(SzYY,2) - SzzYY*SYY) + SzzYY*pow(SXY,2)) - SzzXY*(SXX*(SyzYY*SYY + SxzXY*SYY + (-SyYY - SxXY)*SzYY) + SzXY*(-SyXY*SYY - SxXX*SYY + (SyYY + SxXY)*SXY) + (-SyzYY - SxzXY)*pow(SXY,2) + (SyXY*SzYY + SxXX*SzYY)*SXY) - SzXY*(SzXX*(-SyzYY*SYY - SxzXY*SYY + (SyYY + SxXY)*SzYY) + (-SyXY*SzzYY - SxXX*SzzYY)*SXY - 2*SyzXY*SzYY*SXY - 2*SxzXX*SzYY*SXY + ((SyYY + SxXY)*SzzYY + (-SyzYY - SxzXY)*SzYY)*SXX) - pow(SzXY,2)*(SyzXY*SYY + SxzXX*SYY + (SyzYY + SxzXY)*SXY + SyXY*SzYY + SxXX*SzYY) - (-SyYY - SxXY)*pow(SzXY,3))/denom;
	    double b = (SzzXX*(SyXY*(SzzYY*SYY - pow(SzYY,2)) + SxXX*(SzzYY*SYY - pow(SzYY,2)) + SzXY*(-SyzYY*SYY - SxzXY*SYY + (SyYY + SxXY)*SzYY) + ((-SyYY - SxXY)*SzzYY + (SyzYY + SxzXY)*SzYY)*SXY) + SzXX*(SyzXY*(pow(SzYY,2) - SzzYY*SYY) + SxzXX*(pow(SzYY,2) - SzzYY*SYY)) + SzzXY*(SzXX*(SyzYY*SYY + SxzXY*SYY + (-SyYY - SxXY)*SzYY) + SzXY*(SyzXY*SYY + SxzXX*SYY + (-SyzYY - SxzXY)*SXY + 2*SyXY*SzYY + 2*SxXX*SzYY) - SyzXY*SzYY*SXY - SxzXX*SzYY*SXY + (-SyYY - SxXY)*pow(SzXY,2)) + pow(SzzXY,2)*(-SyXY*SYY - SxXX*SYY + (SyYY + SxXY)*SXY) + SzXY*(SyzXY*SzzYY*SXY + SxzXX*SzzYY*SXY + SzXX*((SyYY + SxXY)*SzzYY + (-SyzYY - SxzXY)*SzYY)) + pow(SzXY,2)*(-SyXY*SzzYY - SxXX*SzzYY - SyzXY*SzYY - SxzXX*SzYY) + (SyzYY + SxzXY)*pow(SzXY,3))/denom;
	    double c = (-SzzXY*(SyzXY*(SXX*SYY - pow(SXY,2)) + SxzXX*(SXX*SYY - pow(SXY,2)) + SzXX*(-SyXY*SYY - SxXX*SYY + (SyYY + SxXY)*SXY) + SzXY*((SyXY + SxXX)*SXY + (-SyYY - SxXY)*SXX)) - SzzXX*(SXX*(-SyzYY*SYY - SxzXY*SYY + (SyYY + SxXY)*SzYY) + SzXY*(SyXY*SYY + SxXX*SYY + (-SyYY - SxXY)*SXY) + (SyzYY + SxzXY)*pow(SXY,2) + (-SyXY*SzYY - SxXX*SzYY)*SXY) - SzXY*(SzXX*(-SyzXY*SYY - SxzXX*SYY + (-2*SyzYY - 2*SxzXY)*SXY + SyXY*SzYY + SxXX*SzYY) - SyzXY*SzYY*SXX - SxzXX*SzYY*SXX) - pow(SzXX,2)*(SyzYY*SYY + SxzXY*SYY + (-SyYY - SxXY)*SzYY) - SzXX*(SyzXY*SzYY*SXY + SxzXX*SzYY*SXY) - pow(SzXY,2)*(SyzXY*SXY + SxzXX*SXY + (SyzYY + SxzXY)*SXX + (SyYY + SxXY)*SzXX) - (-SyXY - SxXX)*pow(SzXY,3))/denom;
	    double d = (SzzXX*(SzXY*((SyzYY + SxzXY)*SXY + SyXY*SzYY + SxXX*SzYY) + (-SyXY*SzzYY - SxXX*SzzYY)*SXY + ((SyYY + SxXY)*SzzYY + (-SyzYY - SxzXY)*SzYY)*SXX + (-SyYY - SxXY)*pow(SzXY,2)) + SzzXY*(SzXX*((-SyzYY - SxzXY)*SXY - SyXY*SzYY - SxXX*SzYY) + SzXY*(-SyzXY*SXY - SxzXX*SXY + (SyzYY + SxzXY)*SXX + (2*SyYY + 2*SxXY)*SzXX) + SyzXY*SzYY*SXX + SxzXX*SzYY*SXX + (-SyXY - SxXX)*pow(SzXY,2)) + SzXX*(SyzXY*SzzYY*SXY + SxzXX*SzzYY*SXY) + pow(SzzXY,2)*((SyXY + SxXX)*SXY + (-SyYY - SxXY)*SXX) + SzXY*(-SyzXY*SzzYY*SXX - SxzXX*SzzYY*SXX + SzXX*(SyXY*SzzYY + SxXX*SzzYY - SyzXY*SzYY - SxzXX*SzYY)) + pow(SzXX,2)*((-SyYY - SxXY)*SzzYY + (SyzYY + SxzXY)*SzYY) + (SyzXY + SxzXX)*pow(SzXY,3) + (-SyzYY - SxzXY)*SzXX*pow(SzXY,2))/denom;

	    GlobalVector momentum = chamberSurface->toGlobal(LocalVector(a, c, 1.) / sqrt(pow(a,2) + pow(c,2) + 1.));
	    if (m_debuggingPrintouts) std::cout << "Momentum direction is " << momentum << std::endl;

	    if (m_debuggingPrintouts) {
	       for (std::vector<const TrackingRecHit*>::const_iterator hit = station->begin();  hit != station->end();  ++hit) {
		  CSCDetId id((*hit)->geographicalId());
		  if (id.chamber() % 2 == bit) {  // if it equals bit, fit!
		     LocalPoint localPoint = (*hit)->localPosition();
		     const Surface& layerSurface = cscGeometry->idToDet(id)->surface();
		     LocalPoint chamberPoint = chamberSurface->toLocal(layerSurface.toGlobal(localPoint));

		     double x = a * chamberPoint.z() + b;
		     double y = c * chamberPoint.z() + d;
		     std::cout << "    fitted hit on " << id << " has hit-minus-track residual (" << (chamberPoint.x() - x) << ", " << (chamberPoint.y() - y) << ")" << std::endl;
		  }
	       }
	    } // end debuggingPrintouts

	    // collecting hits for the alignment
	    for (std::vector<const TrackingRecHit*>::const_iterator hit = station->begin();  hit != station->end();  ++hit) {
	       CSCDetId id((*hit)->geographicalId());
	       if (id.chamber() % 2 != bit) {  // if it doesn't equal bit, don't fit!
	       
		  TrajectoryMeasurement::ConstRecHitPointer hitPtr(muonTransBuilder.build(&**hit, globalGeometry));

		  LocalPoint localPoint = (*hit)->localPosition();
		  AlgebraicSymMatrix localErrorWithAPE = hitPtr->parametersError();

		  const Surface& layerSurface = cscGeometry->idToDet(id)->surface();
		  LocalPoint chamberPoint = chamberSurface->toLocal(layerSurface.toGlobal(localPoint));

		  // the actual track prediction
		  double x = a * chamberPoint.z() + b;
		  double y = c * chamberPoint.z() + d;

		  if (m_debuggingPrintouts) std::cout << "    alignable hit on " << id << " has hit-minus-track residual (" << (chamberPoint.x() - x) << ", " << (chamberPoint.y() - y) << ")" << std::endl;
		     
		  // you need to know what the residual is (for this little calculation, let's do track-minus-hit
		  double rx = x - chamberPoint.x();
		  double ry = y - chamberPoint.y();

		  // transform (rx, ry) to (rxprime, ryprime), in which ryprime is parallel to the strip
		  int strip = cscGeometry->layer(id)->geometry()->nearestStrip(localPoint);
		  double angle = cscGeometry->layer(id)->geometry()->stripAngle(strip) - M_PI/2.;

		  double rxprime =  cos(angle) * rx + sin(angle) * ry;
		  double ryprime = -sin(angle) * rx + cos(angle) * ry;

		  // project onto the rxprime axis
		  ryprime = 0.;

		  // transform back to (rx, ry)
		  rx = cos(angle) * rxprime - sin(angle) * ryprime;
		  ry = sin(angle) * rxprime + cos(angle) * ryprime;

		  // and make a new predicted track position from the projected residual
		  x = rx + chamberPoint.x();
		  y = ry + chamberPoint.y();

		  if (m_debuggingNtuple) {
		     m_ntuple_endcap = id.endcap();
		     m_ntuple_station = id.station();
		     m_ntuple_ring = id.ring();
		     m_ntuple_chamber = id.chamber();
		     m_ntuple_layer = id.layer();
		     m_ntuple_strip = strip;
		     m_ntuple_trackx = x;
		     m_ntuple_tracky = y;
		     m_ntuple_hitx = chamberPoint.x();
		     m_ntuple_hity = chamberPoint.y();
		     m_ntuple_stripAngle = angle;
		     m_ntuple->Fill();
		  }

		  // now proceed as usual
		  GlobalPoint position = chamberSurface->toGlobal(LocalPoint(x, y, chamberPoint.z()));

		  GlobalTrajectoryParameters globalTrajectoryParameters(position, momentum, track->charge(), &*magneticField);
		  AlgebraicSymMatrix66 error;
		  error(0,0) = 1e-6 * position.x();
		  error(1,1) = 1e-6 * position.y();
		  error(2,2) = 1e-6 * position.z();
		  error(3,3) = 1e-6 * momentum.x();
		  error(4,4) = 1e-6 * momentum.y();
		  error(5,5) = 1e-6 * momentum.z();

		  // these must be in lock-step
		  clonedHits.push_back((*hit)->clone());
		  transHits.push_back(hitPtr);
		  TSOSes.push_back(TrajectoryStateOnSurface(globalTrajectoryParameters, CartesianTrajectoryError(error), cscGeometry->idToDet(id)->surface()));

	       } // end if this is not a hit for fitting
	    } // end loop over hits
	 } // end loop over two fits: one from A to B, the other from B to A
      } // end loop over stations
      assert(clonedHits.size() == transHits.size());
      assert(transHits.size() == TSOSes.size());

      // build the trajectory
      if (clonedHits.size() > 0) {
	 if (m_debuggingPrintouts) std::cout << "Creating " << clonedHits.size() << " track projection/hit pairs to pass to HIPAlignmentAlgorithm" << std::endl;

	 PTrajectoryStateOnDet *PTraj = transformer.persistentState(*(TSOSes.begin()), clonedHits.begin()->geographicalId().rawId());
	 TrajectorySeed trajectorySeed(*PTraj, clonedHits, alongMomentum);
	 Trajectory trajectory(trajectorySeed, alongMomentum);

	 edm::OwnVector<TrackingRecHit>::const_iterator clonedHit = clonedHits.begin();
	 std::vector<TrajectoryMeasurement::ConstRecHitPointer>::const_iterator transHitPtr = transHits.begin();
	 std::vector<TrajectoryStateOnSurface>::const_iterator TSOS = TSOSes.begin();
	 for (;  clonedHit != clonedHits.end();  ++clonedHit, ++transHitPtr, ++TSOS) {
	    trajectory.push(TrajectoryMeasurement(*TSOS, *TSOS, *TSOS, (*transHitPtr)));
	 }

	 trajectoryCollection->push_back(trajectory);

	 // Remember which Trajectory is associated with which Track
	 trajCounter++;
	 reference_map[trajCounter] = trackCounter;

      } // end if there are any clonedHits/TSOSes to work with

      if (m_debuggingPrintouts) std::cout << "Done with track" << std::endl;
   } // end loop over tracks

   unsigned int numTrajectories = trajectoryCollection->size();

   // insert the trajectories into the Event
   edm::OrphanHandle<std::vector<Trajectory> > ohTrajs = iEvent.put(trajectoryCollection);

   // create the trajectory <-> track association map
   std::auto_ptr<TrajTrackAssociationCollection> trajTrackMap(new TrajTrackAssociationCollection());

   for (trajCounter = 0;  trajCounter < numTrajectories;  trajCounter++) {
      edm::Ref<reco::TrackCollection>::key_type trackCounter = reference_map[trajCounter];

      trajTrackMap->insert(edm::Ref<std::vector<Trajectory> >(ohTrajs, trajCounter), edm::Ref<reco::TrackCollection>(tracks, trackCounter));
   }
   // and put it in the Event, also
   iEvent.put(trajTrackMap);

   if (m_debuggingPrintouts) std::cout << "Done with event" << std::endl;
}

// ------------ method called once each job just before starting event loop  ------------
void 
MuonHIPOverlapsStripwiseRefitter::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
MuonHIPOverlapsStripwiseRefitter::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(MuonHIPOverlapsStripwiseRefitter);
