// -*- C++ -*-
//
// Package:    ZeroFieldMuonHIPRefitter
// Class:      ZeroFieldMuonHIPRefitter
// 
/**\class ZeroFieldMuonHIPRefitter ZeroFieldMuonHIPRefitter.cc Alignment/ZeroFieldMuonHIPRefitter/src/ZeroFieldMuonHIPRefitter.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jim Pivarski
//         Created:  Wed Dec 12 13:31:55 CST 2007
// $Id: ZeroFieldMuonHIPRefitter.cc,v 1.3 2008/08/22 19:56:42 pivarski Exp $
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
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
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
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"
#include "TTree.h"

// products
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

//
// class decleration
//

class ZeroFieldMuonHIPRefitter : public edm::EDProducer {
   public:
      explicit ZeroFieldMuonHIPRefitter(const edm::ParameterSet&);
      ~ZeroFieldMuonHIPRefitter();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      // ----------member data ---------------------------
      edm::InputTag m_input;
      std::string m_propagator;
      int m_minDOF;
      double m_minRedChi2, m_maxRedChi2;

      bool m_debuggingNtuples;
      TTree *m_tracker;
      Int_t m_tracker_subdetid;
      Float_t m_tracker_hitx, m_tracker_hity, m_tracker_hitz, m_tracker_trackx, m_tracker_tracky, m_tracker_trackz, m_tracker_chi2i;
      Int_t m_tracker_infit, m_tracker_survive, m_tracker_dof;
      Float_t m_tracker_chi2;
      TTree *m_dt;
      Int_t m_dt_wheel, m_dt_station, m_dt_sector, m_dt_superlayer, m_dt_layer;
      Float_t m_dt_hitx, m_dt_hity, m_dt_hitz, m_dt_trackx, m_dt_tracky, m_dt_trackz, m_dt_chi2i;
      Int_t m_dt_survive, m_dt_dof;
      Float_t m_dt_chi2;
      TTree *m_csc;
      Int_t m_csc_station, m_csc_ring, m_csc_chamber, m_csc_layer;
      Float_t m_csc_hitx, m_csc_hity, m_csc_hitz, m_csc_trackx, m_csc_tracky, m_csc_trackz, m_csc_chi2i;
      Int_t m_csc_survive, m_csc_dof;
      Float_t m_csc_chi2;
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
ZeroFieldMuonHIPRefitter::ZeroFieldMuonHIPRefitter(const edm::ParameterSet& iConfig)
   : m_input(iConfig.getParameter<edm::InputTag>("input"))
   , m_propagator(iConfig.getParameter<std::string>("propagator"))
   , m_minDOF(iConfig.getParameter<int>("minDOF"))
   , m_minRedChi2(iConfig.getParameter<double>("minRedChi2"))
   , m_maxRedChi2(iConfig.getParameter<double>("maxRedChi2"))
   , m_debuggingNtuples(iConfig.getUntrackedParameter<bool>("debuggingNtuples", false))
{
   produces<std::vector<Trajectory> >();
   produces<TrajTrackAssociationCollection>();

   m_tracker = m_dt = m_csc = NULL;
   if (m_debuggingNtuples) {
      edm::Service<TFileService> tfile;

      m_tracker = tfile->make<TTree>("tracker", "tracker");
      m_tracker->Branch("subdetid", &m_tracker_subdetid, "subdetid/I");
      m_tracker->Branch("hitx", &m_tracker_hitx, "hitx/F");
      m_tracker->Branch("hity", &m_tracker_hity, "hity/F");
      m_tracker->Branch("hitz", &m_tracker_hitz, "hitz/F");
      m_tracker->Branch("trackx", &m_tracker_trackx, "trackx/F");
      m_tracker->Branch("tracky", &m_tracker_tracky, "tracky/F");
      m_tracker->Branch("trackz", &m_tracker_trackz, "trackz/F");
      m_tracker->Branch("chi2i", &m_tracker_chi2i, "chi2i/F");
      m_tracker->Branch("infit", &m_tracker_infit, "infit/I");
      m_tracker->Branch("survive", &m_tracker_survive, "survive/I");
      m_tracker->Branch("dof", &m_tracker_dof, "dof/I");
      m_tracker->Branch("chi2", &m_tracker_chi2, "chi2");

      m_dt = tfile->make<TTree>("dt", "dt");
      m_dt->Branch("wheel", &m_dt_wheel, "wheel/I");
      m_dt->Branch("station", &m_dt_station, "station/I");
      m_dt->Branch("sector", &m_dt_sector, "sector/I");
      m_dt->Branch("superlayer", &m_dt_superlayer, "superlayer/I");
      m_dt->Branch("layer", &m_dt_layer, "layer/I");
      m_dt->Branch("hitx", &m_dt_hitx, "hitx/F");
      m_dt->Branch("hity", &m_dt_hity, "hity/F");
      m_dt->Branch("hitz", &m_dt_hitz, "hitz/F");
      m_dt->Branch("trackx", &m_dt_trackx, "trackx/F");
      m_dt->Branch("tracky", &m_dt_tracky, "tracky/F");
      m_dt->Branch("trackz", &m_dt_trackz, "trackz/F");
      m_dt->Branch("chi2i", &m_dt_chi2i, "chi2i/F");
      m_dt->Branch("survive", &m_dt_survive, "survive/I");
      m_dt->Branch("dof", &m_dt_dof, "dof/I");
      m_dt->Branch("chi2", &m_dt_chi2, "chi2");

      m_csc = tfile->make<TTree>("csc", "csc");
      m_csc->Branch("station", &m_csc_station, "station/I");
      m_csc->Branch("ring", &m_csc_ring, "ring/I");
      m_csc->Branch("chamber", &m_csc_chamber, "chamber/I");
      m_csc->Branch("layer", &m_csc_layer, "layer/I");
      m_csc->Branch("hitx", &m_csc_hitx, "hitx/F");
      m_csc->Branch("hity", &m_csc_hity, "hity/F");
      m_csc->Branch("hitz", &m_csc_hitz, "hitz/F");
      m_csc->Branch("trackx", &m_csc_trackx, "trackx/F");
      m_csc->Branch("tracky", &m_csc_tracky, "tracky/F");
      m_csc->Branch("trackz", &m_csc_trackz, "trackz/F");
      m_csc->Branch("chi2i", &m_csc_chi2i, "chi2i/F");
      m_csc->Branch("survive", &m_csc_survive, "survive/I");
      m_csc->Branch("dof", &m_csc_dof, "dof/I");
      m_csc->Branch("chi2", &m_csc_chi2, "chi2");
   }
}


ZeroFieldMuonHIPRefitter::~ZeroFieldMuonHIPRefitter()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
ZeroFieldMuonHIPRefitter::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   edm::Handle<reco::TrackCollection> tracks;
   iEvent.getByLabel(m_input, tracks);

   edm::ESHandle<TrackerGeometry> trackerGeometry;
   iSetup.get<TrackerDigiGeometryRecord>().get(trackerGeometry);

   edm::ESHandle<DTGeometry> dtGeometry;
   iSetup.get<MuonGeometryRecord>().get(dtGeometry);

   edm::ESHandle<CSCGeometry> cscGeometry;
   iSetup.get<MuonGeometryRecord>().get(cscGeometry);

   edm::ESHandle<Propagator> propagator;
   iSetup.get<TrackingComponentsRecord>().get(m_propagator, propagator);

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

   // We only use one track, because the track-finder finds duplicates
   reco::TrackCollection::const_iterator best_track;
   int most_muonhits = 0;
   for (reco::TrackCollection::const_iterator track = tracks->begin();  track != tracks->end();  ++track) {
      int muonhits = 0;
      for (trackingRecHit_iterator hit = track->recHitsBegin();  hit != track->recHitsEnd();  ++hit) {
	 DetId id = (*hit)->geographicalId();
	 if (id.det() == DetId::Muon  &&  (id.subdetId() == MuonSubdetId::DT  ||  id.subdetId() == MuonSubdetId::CSC)) {
	    muonhits++;
	 }
      }
      if (muonhits > most_muonhits) {
	 best_track = track;
	 most_muonhits = muonhits;
      }
   }

   // these must be in lock-step
   edm::OwnVector<TrackingRecHit> clonedHits;
   std::vector<TrajectoryMeasurement::ConstRecHitPointer> transHits;
   std::vector<TrajectoryStateOnSurface> TSOSes;

   // fit once with all the PXB/TIB/TOB/DT hits, then with all the PXF/TID/TEC/CSC hits; many tracks won't have enough of both to fit twice
   for (int radial = 0;  radial <= 1;  radial++) {
      // we want the independent variable of the fit to be roughly collinear with the resulting line
      // (so the final fitted slope will be a small correction)
      // we convert from local to global, and then left-multiply by this rotation
      math::XYZVector direction = best_track->momentum();
      double theta1 = atan2(-direction.x(), direction.y());
      double theta2 = atan2(direction.z(), sqrt(direction.perp2()));
      align::RotationType coordrot(             cos(theta1),  sin(theta1),             0.,
				    sin(theta1)*sin(theta2), -cos(theta1)*sin(theta2), cos(theta2),
				   -sin(theta1)*cos(theta2),  cos(theta1)*cos(theta2), sin(theta2));

      if (radial) {
	 theta2 = 0.;
	 coordrot = align::RotationType(             cos(theta1),  sin(theta1),             0.,
					 sin(theta1)*sin(theta2), -cos(theta1)*sin(theta2), cos(theta2),
					-sin(theta1)*cos(theta2),  cos(theta1)*cos(theta2), sin(theta2));
      }
      else {
	 coordrot = align::RotationType(1., 0., 0.,
					0., 1., 0.,
					0., 0., 1.);
      }

      // these x and y values are perpendicular to the direction-of-momentum axis
      double SXX, SxXX, SxXY, SXY, SxzXX, SxzXY, SyXY, SYY, SyYY, SyzXY, SyzYY, SzXX, SzXY, SzYY, SzzXX, SzzXY, SzzYY;
      SXX = SxXX = SxXY = SXY = SxzXX = SxzXY = SyXY = SYY = SyYY = SyzXY = SyzYY = SzXX = SzXY = SzYY = SzzXX = SzzXY = SzzYY = 0.;
      double sumz = 0.;
      double numz = 0.;

      for (trackingRecHit_iterator hit = best_track->recHitsBegin();  hit != best_track->recHitsEnd();  ++hit) {
	 DetId id = (*hit)->geographicalId();
	 if (id.det() == DetId::Tracker) {
	    if ((radial  &&  (id.subdetId() == PixelSubdetector::PixelBarrel  ||  id.subdetId() == StripSubdetector::TIB  ||  id.subdetId() == StripSubdetector::TOB))  ||
		(!radial  &&  (id.subdetId() == PixelSubdetector::PixelEndcap  ||  id.subdetId() == StripSubdetector::TID  ||  id.subdetId() == StripSubdetector::TEC))) {

	       LocalPoint localPoint = (*hit)->localPosition();
	       LocalError localError = (*hit)->localPositionError();

	       const Surface& surface = trackerGeometry->idToDet(id)->surface();
	 
	       GlobalPoint position = surface.toGlobal(localPoint);
	       position = GlobalPoint(coordrot.xx()*position.x() + coordrot.xy()*position.y() + coordrot.xz()*position.z(),
				      coordrot.yx()*position.x() + coordrot.yy()*position.y() + coordrot.yz()*position.z(),
				      coordrot.zx()*position.x() + coordrot.zy()*position.y() + coordrot.zz()*position.z());
	 
	       align::RotationType rotation = coordrot * surface.rotation().transposed();
	       align::RotationType errorAsRotation = rotation * align::RotationType(localError.xx(), localError.xy(), 0, localError.xy(), localError.yy(), 0, 0, 0, 0) * rotation.transposed();

	       AlgebraicSymMatrix error(2);
	       error[0][0] = errorAsRotation.xx();
	       error[1][1] = errorAsRotation.yy();
	       error[0][1] = errorAsRotation.yx();

	       int ierr;
	       error.invert(ierr);
	       if (ierr != 0) {
		  edm::LogError("ZeroFieldMuonHIPRefitter") << "Matrix inversion failed!  ierr = " << ierr << " matrix = " << std::endl << error << std::endl;
		  return;
	       }

	       // these x and y values are perpendicular to the direction-of-momentum axis
	       double xi = position.x();
	       double yi = position.y();
	       double zi = position.z();
	       double XX = error[0][0];
	       double XY = error[0][1];
	       double YY = error[1][1];
	       sumz += zi;
	       numz += 1.;

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

	    } // end if the right kind of tracker hit (for radial or non-radial)
	 } // end if tracker hit
      } // end loop over hits

      // calculate the least-squares fit
      double denom = (SzzXX*(SXX*(SzzYY*SYY - pow(SzYY,2)) - pow(SzXY,2)*SYY - SzzYY*pow(SXY,2) + 2*SzXY*SzYY*SXY) + SzzXY*(SzXY*(2*SzXX*SYY + 2*SzYY*SXX) - 2*SzXX*SzYY*SXY - 2*pow(SzXY,2)*SXY) + pow(SzzXY,2)*(pow(SXY,2) - SXX*SYY) + pow(SzXX,2)*(pow(SzYY,2) - SzzYY*SYY) + 2*SzXX*SzXY*SzzYY*SXY + pow(SzXY,2)*(-SzzYY*SXX - 2*SzXX*SzYY) + pow(SzXY,4));

      if (denom != 0.  &&  numz > 0.) {
	 double a = (-SzXX*(SyXY*(SzzYY*SYY - pow(SzYY,2)) + SxXX*(SzzYY*SYY - pow(SzYY,2)) + ((-SyYY - SxXY)*SzzYY + (SyzYY + SxzXY)*SzYY)*SXY) - SyzXY*(SXX*(pow(SzYY,2) - SzzYY*SYY) + SzzYY*pow(SXY,2)) - SxzXX*(SXX*(pow(SzYY,2) - SzzYY*SYY) + SzzYY*pow(SXY,2)) - SzzXY*(SXX*(SyzYY*SYY + SxzXY*SYY + (-SyYY - SxXY)*SzYY) + SzXY*(-SyXY*SYY - SxXX*SYY + (SyYY + SxXY)*SXY) + (-SyzYY - SxzXY)*pow(SXY,2) + (SyXY*SzYY + SxXX*SzYY)*SXY) - SzXY*(SzXX*(-SyzYY*SYY - SxzXY*SYY + (SyYY + SxXY)*SzYY) + (-SyXY*SzzYY - SxXX*SzzYY)*SXY - 2*SyzXY*SzYY*SXY - 2*SxzXX*SzYY*SXY + ((SyYY + SxXY)*SzzYY + (-SyzYY - SxzXY)*SzYY)*SXX) - pow(SzXY,2)*(SyzXY*SYY + SxzXX*SYY + (SyzYY + SxzXY)*SXY + SyXY*SzYY + SxXX*SzYY) - (-SyYY - SxXY)*pow(SzXY,3))/denom;
	 double b = (SzzXX*(SyXY*(SzzYY*SYY - pow(SzYY,2)) + SxXX*(SzzYY*SYY - pow(SzYY,2)) + SzXY*(-SyzYY*SYY - SxzXY*SYY + (SyYY + SxXY)*SzYY) + ((-SyYY - SxXY)*SzzYY + (SyzYY + SxzXY)*SzYY)*SXY) + SzXX*(SyzXY*(pow(SzYY,2) - SzzYY*SYY) + SxzXX*(pow(SzYY,2) - SzzYY*SYY)) + SzzXY*(SzXX*(SyzYY*SYY + SxzXY*SYY + (-SyYY - SxXY)*SzYY) + SzXY*(SyzXY*SYY + SxzXX*SYY + (-SyzYY - SxzXY)*SXY + 2*SyXY*SzYY + 2*SxXX*SzYY) - SyzXY*SzYY*SXY - SxzXX*SzYY*SXY + (-SyYY - SxXY)*pow(SzXY,2)) + pow(SzzXY,2)*(-SyXY*SYY - SxXX*SYY + (SyYY + SxXY)*SXY) + SzXY*(SyzXY*SzzYY*SXY + SxzXX*SzzYY*SXY + SzXX*((SyYY + SxXY)*SzzYY + (-SyzYY - SxzXY)*SzYY)) + pow(SzXY,2)*(-SyXY*SzzYY - SxXX*SzzYY - SyzXY*SzYY - SxzXX*SzYY) + (SyzYY + SxzXY)*pow(SzXY,3))/denom;
	 double c = (-SzzXY*(SyzXY*(SXX*SYY - pow(SXY,2)) + SxzXX*(SXX*SYY - pow(SXY,2)) + SzXX*(-SyXY*SYY - SxXX*SYY + (SyYY + SxXY)*SXY) + SzXY*((SyXY + SxXX)*SXY + (-SyYY - SxXY)*SXX)) - SzzXX*(SXX*(-SyzYY*SYY - SxzXY*SYY + (SyYY + SxXY)*SzYY) + SzXY*(SyXY*SYY + SxXX*SYY + (-SyYY - SxXY)*SXY) + (SyzYY + SxzXY)*pow(SXY,2) + (-SyXY*SzYY - SxXX*SzYY)*SXY) - SzXY*(SzXX*(-SyzXY*SYY - SxzXX*SYY + (-2*SyzYY - 2*SxzXY)*SXY + SyXY*SzYY + SxXX*SzYY) - SyzXY*SzYY*SXX - SxzXX*SzYY*SXX) - pow(SzXX,2)*(SyzYY*SYY + SxzXY*SYY + (-SyYY - SxXY)*SzYY) - SzXX*(SyzXY*SzYY*SXY + SxzXX*SzYY*SXY) - pow(SzXY,2)*(SyzXY*SXY + SxzXX*SXY + (SyzYY + SxzXY)*SXX + (SyYY + SxXY)*SzXX) - (-SyXY - SxXX)*pow(SzXY,3))/denom;
	 double d = (SzzXX*(SzXY*((SyzYY + SxzXY)*SXY + SyXY*SzYY + SxXX*SzYY) + (-SyXY*SzzYY - SxXX*SzzYY)*SXY + ((SyYY + SxXY)*SzzYY + (-SyzYY - SxzXY)*SzYY)*SXX + (-SyYY - SxXY)*pow(SzXY,2)) + SzzXY*(SzXX*((-SyzYY - SxzXY)*SXY - SyXY*SzYY - SxXX*SzYY) + SzXY*(-SyzXY*SXY - SxzXX*SXY + (SyzYY + SxzXY)*SXX + (2*SyYY + 2*SxXY)*SzXX) + SyzXY*SzYY*SXX + SxzXX*SzYY*SXX + (-SyXY - SxXX)*pow(SzXY,2)) + SzXX*(SyzXY*SzzYY*SXY + SxzXX*SzzYY*SXY) + pow(SzzXY,2)*((SyXY + SxXX)*SXY + (-SyYY - SxXY)*SXX) + SzXY*(-SyzXY*SzzYY*SXX - SxzXX*SzzYY*SXX + SzXX*(SyXY*SzzYY + SxXX*SzzYY - SyzXY*SzYY - SxzXX*SzYY)) + pow(SzXX,2)*((-SyYY - SxXY)*SzzYY + (SyzYY + SxzXY)*SzYY) + (SyzXY + SxzXX)*pow(SzXY,3) + (-SyzYY - SxzXY)*SzXX*pow(SzXY,2))/denom;

	 double averagez = sumz/numz;

	 GlobalVector fitslope = GlobalVector(a, c, 1.) / sqrt(pow(a,2) + pow(c,2) + 1.);
	 GlobalPoint fitpoint = GlobalPoint(a * averagez + b, c * averagez + d, averagez);

	 // left-multiply by the inverse rotation
	 fitslope = GlobalVector(coordrot.xx() * fitslope.x() + coordrot.yx() * fitslope.y() + coordrot.zx() * fitslope.z(),
				 coordrot.xy() * fitslope.x() + coordrot.yy() * fitslope.y() + coordrot.zy() * fitslope.z(),
				 coordrot.xz() * fitslope.x() + coordrot.yz() * fitslope.y() + coordrot.zz() * fitslope.z());
	 fitpoint = GlobalPoint(coordrot.xx() * fitpoint.x() + coordrot.yx() * fitpoint.y() + coordrot.zx() * fitpoint.z(),
				coordrot.xy() * fitpoint.x() + coordrot.yy() * fitpoint.y() + coordrot.zy() * fitpoint.z(),
				coordrot.xz() * fitpoint.x() + coordrot.yz() * fitpoint.y() + coordrot.zz() * fitpoint.z());

	 FreeTrajectoryState freeTrajectoryState(GlobalTrajectoryParameters(fitpoint, fitslope, best_track->charge(), &*magneticField));

// 	 /////////////////////////////////////////////////////////////////////////////////////////////////////////
// 	 /////////// second fit, dropping very bad hits
// 	 // Yes, this is a horrible way to code.  Remember, all of this is temporary, for vetting the official fitter.

// 	 sumz = numz = SXX = SxXX = SxXY = SXY = SxzXX = SxzXY = SyXY = SYY = SyYY = SyzXY = SyzYY = SzXX = SzXY = SzYY = SzzXX = SzzXY = SzzYY = 0.;

// 	 for (trackingRecHit_iterator hit = best_track->recHitsBegin();  hit != best_track->recHitsEnd();  ++hit) {
// 	    DetId id = (*hit)->geographicalId();
// 	    if (id.det() == DetId::Tracker) {
// 	       if ((radial  &&  (id.subdetId() == PixelSubdetector::PixelBarrel  ||  id.subdetId() == StripSubdetector::TIB  ||  id.subdetId() == StripSubdetector::TOB))  ||
// 		   (!radial  &&  (id.subdetId() == PixelSubdetector::PixelEndcap  ||  id.subdetId() == StripSubdetector::TID  ||  id.subdetId() == StripSubdetector::TEC))) {

// 		  LocalPoint localPoint = (*hit)->localPosition();
// 		  LocalError localError = (*hit)->localPositionError();

// 		  const Surface& surface = trackerGeometry->idToDet(id)->surface();
	 
// 		  GlobalPoint position = surface.toGlobal(localPoint);
// 		  position = GlobalPoint(coordrot.xx()*position.x() + coordrot.xy()*position.y() + coordrot.xz()*position.z(),
// 					 coordrot.yx()*position.x() + coordrot.yy()*position.y() + coordrot.yz()*position.z(),
// 					 coordrot.zx()*position.x() + coordrot.zy()*position.y() + coordrot.zz()*position.z());
	 
// 		  align::RotationType rotation = coordrot * surface.rotation().transposed();
// 		  align::RotationType errorAsRotation = rotation * align::RotationType(localError.xx(), localError.xy(), 0, localError.xy(), localError.yy(), 0, 0, 0, 0) * rotation.transposed();

// 		  AlgebraicSymMatrix error(2);
// 		  error[0][0] = errorAsRotation.xx();
// 		  error[1][1] = errorAsRotation.yy();
// 		  error[0][1] = errorAsRotation.yx();

// 		  int ierr;
// 		  error.invert(ierr);
// 		  if (ierr != 0) {
// 		     edm::LogError("ZeroFieldMuonHIPRefitter") << "Matrix inversion failed!  ierr = " << ierr << " matrix = " << std::endl << error << std::endl;
// 		     return;
// 		  }

// 		  // these x and y values are perpendicular to the direction-of-momentum axis
// 		  double xi = position.x();
// 		  double yi = position.y();
// 		  double zi = position.z();
// 		  double XX = error[0][0];
// 		  double XY = error[0][1];
// 		  double YY = error[1][1];

// 		  TrajectoryStateOnSurface trackTSOS = propagator->propagate(freeTrajectoryState, surface);
// 		  double tx = trackTSOS.localPosition().x();
// 		  double ty = trackTSOS.localPosition().y();
// 		  double hx = (*hit)->localPosition().x();
// 		  double hy = (*hit)->localPosition().y();

// 		  AlgebraicSymMatrix covmat(2);
// 		  covmat[0][0] = (*hit)->localPositionError().xx();
// 		  covmat[0][1] = (*hit)->localPositionError().xy();
// 		  covmat[1][1] = (*hit)->localPositionError().yy();

// 		  double chi2i;
// 		  if ((*hit)->dimension() == 1) {
// 		     chi2i = (tx - hx)*(tx - hx) / covmat[0][0];  // note the division (like an inverse)
// 		  }
// 		  else {
// 		     int ierr;
// 		     covmat.invert(ierr);
// 		     if (ierr != 0) {
// 			edm::LogError("ZeroFieldMuonHIPRefitter") << "Matrix inversion failed (muon)!  ierr = " << ierr << " matrix = " << std::endl << covmat << std::endl;
// 			return;
// 		     }

// 		     chi2i = (tx - hx)*(tx - hx)*covmat[0][0] + 2.*(tx - hx)*(ty - hy)*covmat[0][1] + (ty - hy)*(ty - hy)*covmat[1][1];
// 		  }
		  
// 		  if (chi2i < 10.) {
// 		     sumz += zi;
// 		     numz += 1.;

// 		     SXX += XX;
// 		     SxXX += xi * XX;
// 		     SxXY += xi * XY;
// 		     SXY += XY;
// 		     SxzXX += xi * zi * XX;
// 		     SxzXY += xi * zi * XY;
// 		     SyXY += yi * XY;
// 		     SYY += YY;
// 		     SyYY += yi * YY;
// 		     SyzXY += yi * zi * XY;
// 		     SyzYY += yi * zi * YY;
// 		     SzXX += zi * XX;
// 		     SzXY += zi * XY;
// 		     SzYY += zi * YY;
// 		     SzzXX += zi * zi * XX;
// 		     SzzXY += zi * zi * XY;
// 		     SzzYY += zi * zi * YY;
// 		  } // end if hit is not very bad

// 	       } // end if the right kind of tracker hit (for radial or non-radial)
// 	    } // end if tracker hit
// 	 } // end loop over hits

// 	 denom = (SzzXX*(SXX*(SzzYY*SYY - pow(SzYY,2)) - pow(SzXY,2)*SYY - SzzYY*pow(SXY,2) + 2*SzXY*SzYY*SXY) + SzzXY*(SzXY*(2*SzXX*SYY + 2*SzYY*SXX) - 2*SzXX*SzYY*SXY - 2*pow(SzXY,2)*SXY) + pow(SzzXY,2)*(pow(SXY,2) - SXX*SYY) + pow(SzXX,2)*(pow(SzYY,2) - SzzYY*SYY) + 2*SzXX*SzXY*SzzYY*SXY + pow(SzXY,2)*(-SzzYY*SXX - 2*SzXX*SzYY) + pow(SzXY,4));

// 	 if (denom != 0.  &&  numz > 0.) {
// 	    a = (-SzXX*(SyXY*(SzzYY*SYY - pow(SzYY,2)) + SxXX*(SzzYY*SYY - pow(SzYY,2)) + ((-SyYY - SxXY)*SzzYY + (SyzYY + SxzXY)*SzYY)*SXY) - SyzXY*(SXX*(pow(SzYY,2) - SzzYY*SYY) + SzzYY*pow(SXY,2)) - SxzXX*(SXX*(pow(SzYY,2) - SzzYY*SYY) + SzzYY*pow(SXY,2)) - SzzXY*(SXX*(SyzYY*SYY + SxzXY*SYY + (-SyYY - SxXY)*SzYY) + SzXY*(-SyXY*SYY - SxXX*SYY + (SyYY + SxXY)*SXY) + (-SyzYY - SxzXY)*pow(SXY,2) + (SyXY*SzYY + SxXX*SzYY)*SXY) - SzXY*(SzXX*(-SyzYY*SYY - SxzXY*SYY + (SyYY + SxXY)*SzYY) + (-SyXY*SzzYY - SxXX*SzzYY)*SXY - 2*SyzXY*SzYY*SXY - 2*SxzXX*SzYY*SXY + ((SyYY + SxXY)*SzzYY + (-SyzYY - SxzXY)*SzYY)*SXX) - pow(SzXY,2)*(SyzXY*SYY + SxzXX*SYY + (SyzYY + SxzXY)*SXY + SyXY*SzYY + SxXX*SzYY) - (-SyYY - SxXY)*pow(SzXY,3))/denom;
// 	    b = (SzzXX*(SyXY*(SzzYY*SYY - pow(SzYY,2)) + SxXX*(SzzYY*SYY - pow(SzYY,2)) + SzXY*(-SyzYY*SYY - SxzXY*SYY + (SyYY + SxXY)*SzYY) + ((-SyYY - SxXY)*SzzYY + (SyzYY + SxzXY)*SzYY)*SXY) + SzXX*(SyzXY*(pow(SzYY,2) - SzzYY*SYY) + SxzXX*(pow(SzYY,2) - SzzYY*SYY)) + SzzXY*(SzXX*(SyzYY*SYY + SxzXY*SYY + (-SyYY - SxXY)*SzYY) + SzXY*(SyzXY*SYY + SxzXX*SYY + (-SyzYY - SxzXY)*SXY + 2*SyXY*SzYY + 2*SxXX*SzYY) - SyzXY*SzYY*SXY - SxzXX*SzYY*SXY + (-SyYY - SxXY)*pow(SzXY,2)) + pow(SzzXY,2)*(-SyXY*SYY - SxXX*SYY + (SyYY + SxXY)*SXY) + SzXY*(SyzXY*SzzYY*SXY + SxzXX*SzzYY*SXY + SzXX*((SyYY + SxXY)*SzzYY + (-SyzYY - SxzXY)*SzYY)) + pow(SzXY,2)*(-SyXY*SzzYY - SxXX*SzzYY - SyzXY*SzYY - SxzXX*SzYY) + (SyzYY + SxzXY)*pow(SzXY,3))/denom;
// 	    c = (-SzzXY*(SyzXY*(SXX*SYY - pow(SXY,2)) + SxzXX*(SXX*SYY - pow(SXY,2)) + SzXX*(-SyXY*SYY - SxXX*SYY + (SyYY + SxXY)*SXY) + SzXY*((SyXY + SxXX)*SXY + (-SyYY - SxXY)*SXX)) - SzzXX*(SXX*(-SyzYY*SYY - SxzXY*SYY + (SyYY + SxXY)*SzYY) + SzXY*(SyXY*SYY + SxXX*SYY + (-SyYY - SxXY)*SXY) + (SyzYY + SxzXY)*pow(SXY,2) + (-SyXY*SzYY - SxXX*SzYY)*SXY) - SzXY*(SzXX*(-SyzXY*SYY - SxzXX*SYY + (-2*SyzYY - 2*SxzXY)*SXY + SyXY*SzYY + SxXX*SzYY) - SyzXY*SzYY*SXX - SxzXX*SzYY*SXX) - pow(SzXX,2)*(SyzYY*SYY + SxzXY*SYY + (-SyYY - SxXY)*SzYY) - SzXX*(SyzXY*SzYY*SXY + SxzXX*SzYY*SXY) - pow(SzXY,2)*(SyzXY*SXY + SxzXX*SXY + (SyzYY + SxzXY)*SXX + (SyYY + SxXY)*SzXX) - (-SyXY - SxXX)*pow(SzXY,3))/denom;
// 	    d = (SzzXX*(SzXY*((SyzYY + SxzXY)*SXY + SyXY*SzYY + SxXX*SzYY) + (-SyXY*SzzYY - SxXX*SzzYY)*SXY + ((SyYY + SxXY)*SzzYY + (-SyzYY - SxzXY)*SzYY)*SXX + (-SyYY - SxXY)*pow(SzXY,2)) + SzzXY*(SzXX*((-SyzYY - SxzXY)*SXY - SyXY*SzYY - SxXX*SzYY) + SzXY*(-SyzXY*SXY - SxzXX*SXY + (SyzYY + SxzXY)*SXX + (2*SyYY + 2*SxXY)*SzXX) + SyzXY*SzYY*SXX + SxzXX*SzYY*SXX + (-SyXY - SxXX)*pow(SzXY,2)) + SzXX*(SyzXY*SzzYY*SXY + SxzXX*SzzYY*SXY) + pow(SzzXY,2)*((SyXY + SxXX)*SXY + (-SyYY - SxXY)*SXX) + SzXY*(-SyzXY*SzzYY*SXX - SxzXX*SzzYY*SXX + SzXX*(SyXY*SzzYY + SxXX*SzzYY - SyzXY*SzYY - SxzXX*SzYY)) + pow(SzXX,2)*((-SyYY - SxXY)*SzzYY + (SyzYY + SxzXY)*SzYY) + (SyzXY + SxzXX)*pow(SzXY,3) + (-SyzYY - SxzXY)*SzXX*pow(SzXY,2))/denom;

// 	    averagez = sumz/numz;

// 	    fitslope = GlobalVector(a, c, 1.) / sqrt(pow(a,2) + pow(c,2) + 1.);
// 	    fitpoint = GlobalPoint(a * averagez + b, c * averagez + d, averagez);

// 	    // left-multiply by the inverse rotation
// 	    fitslope = GlobalVector(coordrot.xx() * fitslope.x() + coordrot.yx() * fitslope.y() + coordrot.zx() * fitslope.z(),
// 				    coordrot.xy() * fitslope.x() + coordrot.yy() * fitslope.y() + coordrot.zy() * fitslope.z(),
// 				    coordrot.xz() * fitslope.x() + coordrot.yz() * fitslope.y() + coordrot.zz() * fitslope.z());
// 	    fitpoint = GlobalPoint(coordrot.xx() * fitpoint.x() + coordrot.yx() * fitpoint.y() + coordrot.zx() * fitpoint.z(),
// 				   coordrot.xy() * fitpoint.x() + coordrot.yy() * fitpoint.y() + coordrot.zy() * fitpoint.z(),
// 				   coordrot.xz() * fitpoint.x() + coordrot.yz() * fitpoint.y() + coordrot.zz() * fitpoint.z());

// 	    freeTrajectoryState = FreeTrajectoryState(GlobalTrajectoryParameters(fitpoint, fitslope, best_track->charge(), &*magneticField));

// 	 /////////// end second fit
// 	 /////////////////////////////////////////////////////////////////////////////////////////////////////////

	 double chi2 = 0.;
	 int dof = 0;
	 for (trackingRecHit_iterator hit = best_track->recHitsBegin();  hit != best_track->recHitsEnd();  ++hit) {
	    DetId id = (*hit)->geographicalId();
	    if (id.det() == DetId::Tracker) {
	       if ((radial  &&  (id.subdetId() == PixelSubdetector::PixelBarrel  ||  id.subdetId() == StripSubdetector::TIB  ||  id.subdetId() == StripSubdetector::TOB))  ||
		   (!radial  &&  (id.subdetId() == PixelSubdetector::PixelEndcap  ||  id.subdetId() == StripSubdetector::TID  ||  id.subdetId() == StripSubdetector::TEC))) {
		  const Surface &surface = trackerGeometry->idToDet(id)->surface();

		  TrajectoryStateOnSurface trackTSOS = propagator->propagate(freeTrajectoryState, surface);
		  double tx = trackTSOS.localPosition().x();
		  double ty = trackTSOS.localPosition().y();
		  double hx = (*hit)->localPosition().x();
		  double hy = (*hit)->localPosition().y();

		  AlgebraicSymMatrix covmat(2);
		  covmat[0][0] = (*hit)->localPositionError().xx();
		  covmat[0][1] = (*hit)->localPositionError().xy();
		  covmat[1][1] = (*hit)->localPositionError().yy();

		  double chi2i;
		  if ((*hit)->dimension() == 1) {
		     chi2i = (tx - hx)*(tx - hx) / covmat[0][0];  // note the division (like an inverse)
		  }
		  else {
		     int ierr;
		     covmat.invert(ierr);
		     if (ierr != 0) {
			edm::LogError("ZeroFieldMuonHIPRefitter") << "Matrix inversion failed (muon)!  ierr = " << ierr << " matrix = " << std::endl << covmat << std::endl;
			return;
		     }
		     
		     chi2i = (tx - hx)*(tx - hx)*covmat[0][0] + 2.*(tx - hx)*(ty - hy)*covmat[0][1] + (ty - hy)*(ty - hy)*covmat[1][1];
		  }

		  chi2 += chi2i;
		  if ((*hit)->dimension() == 1) dof += 1;
		  else dof += 2;
	       } // end if the right kind of tracker hit (for radial or non-radial)
	    } // end if tracker hit
	 } // end loop over hits
	 dof -= 4;

	 for (trackingRecHit_iterator hit = best_track->recHitsBegin();  hit != best_track->recHitsEnd();  ++hit) {
	    DetId id = (*hit)->geographicalId();
	    const Surface *surface = NULL;
	    if (id.det() == DetId::Tracker) surface = &(trackerGeometry->idToDet(id)->surface());
	    else if (id.det() == DetId::Muon  &&  id.subdetId() == MuonSubdetId::DT) surface = &(dtGeometry->idToDet(id)->surface());
	    else if (id.det() == DetId::Muon  &&  id.subdetId() == MuonSubdetId::CSC) surface = &(cscGeometry->idToDet(id)->surface());
	    if (surface == NULL) continue;

	    TrajectoryStateOnSurface trackTSOS = propagator->propagate(freeTrajectoryState, *surface);

	    bool survive = (dof >= m_minDOF  &&  chi2 / dof > m_minRedChi2  &&  chi2 / dof < m_maxRedChi2);
	    if (survive  &&  id.det() == DetId::Muon) {
	       GlobalTrajectoryParameters globalTrajectoryParameters(trackTSOS.globalPosition(), trackTSOS.globalMomentum(), best_track->charge(), &*magneticField);
	       AlgebraicSymMatrix66 error;
	       error(0,0) = 1e-6;
	       error(1,1) = 1e-6;
	       error(2,2) = 1e-6;
	       error(3,3) = 1e-6;
	       error(4,4) = 1e-6;
	       error(5,5) = 1e-6;

	       TrajectoryMeasurement::ConstRecHitPointer hitPtr(muonTransBuilder.build(&**hit, globalGeometry));
	       transHits.push_back(hitPtr);
	       clonedHits.push_back((*hit)->clone());
	       TSOSes.push_back(TrajectoryStateOnSurface(globalTrajectoryParameters, CartesianTrajectoryError(error), *surface));
	    }
	    
	    if (m_debuggingNtuples) {
	       double tx = trackTSOS.localPosition().x();
	       double ty = trackTSOS.localPosition().y();
	       double hx = (*hit)->localPosition().x();
	       double hy = (*hit)->localPosition().y();

	       AlgebraicSymMatrix covmat(2);
	       covmat[0][0] = (*hit)->localPositionError().xx();
	       covmat[0][1] = (*hit)->localPositionError().xy();
	       covmat[1][1] = (*hit)->localPositionError().yy();

	       double chi2i;
	       if ((*hit)->dimension() == 1) {
		  chi2i = (tx - hx)*(tx - hx) / covmat[0][0];  // note the division (like an inverse)
	       }
	       else {
		  int ierr;
		  covmat.invert(ierr);
		  if (ierr != 0) {
		     edm::LogError("ZeroFieldMuonHIPRefitter") << "Matrix inversion failed (muon)!  ierr = " << ierr << " matrix = " << std::endl << covmat << std::endl;
		     return;
		  }

		  chi2i = (tx - hx)*(tx - hx)*covmat[0][0] + 2.*(tx - hx)*(ty - hy)*covmat[0][1] + (ty - hy)*(ty - hy)*covmat[1][1];
	       }

	       GlobalPoint hitxyz = surface->toGlobal((*hit)->localPosition());
	       GlobalPoint trackxyz = surface->toGlobal(trackTSOS.localPosition());

	       if (id.det() == DetId::Tracker) {
		  m_tracker_subdetid = id.subdetId();
		  m_tracker_hitx = hitxyz.x();
		  m_tracker_hity = hitxyz.y();
		  m_tracker_hitz = hitxyz.z();
		  m_tracker_trackx = trackxyz.x();
		  m_tracker_tracky = trackxyz.y();
		  m_tracker_trackz = trackxyz.z();
		  m_tracker_chi2i = chi2i;
		  m_tracker_infit = (((radial  &&  (id.subdetId() == PixelSubdetector::PixelBarrel  ||  id.subdetId() == StripSubdetector::TIB  ||  id.subdetId() == StripSubdetector::TOB))  ||
				      (!radial  &&  (id.subdetId() == PixelSubdetector::PixelEndcap  ||  id.subdetId() == StripSubdetector::TID  ||  id.subdetId() == StripSubdetector::TEC))    )
		     );
//				     && (chi2i < 10.));  // for the second-fit option
		  m_tracker_survive = survive;
		  m_tracker_dof = dof;
		  m_tracker_chi2 = chi2;
		  m_tracker->Fill();
	       }
	       else if (id.det() == DetId::Muon  &&  id.subdetId() == MuonSubdetId::DT) {
		  DTLayerId dtId(id.rawId());
		  m_dt_wheel = dtId.wheel();
		  m_dt_station = dtId.station();
		  m_dt_sector = dtId.sector();
		  m_dt_superlayer = dtId.superlayer();
		  m_dt_layer = dtId.layer();
		  m_dt_hitx = hitxyz.x();
		  m_dt_hity = hitxyz.y();
		  m_dt_hitz = hitxyz.z();
		  m_dt_trackx = trackxyz.x();
		  m_dt_tracky = trackxyz.y();
		  m_dt_trackz = trackxyz.z();
		  m_dt_chi2i = chi2i;
		  m_dt_survive = survive;
		  m_dt_dof = dof;
		  m_dt_chi2 = chi2;
		  m_dt->Fill();
	       }
	       else if (id.det() == DetId::Muon  &&  id.subdetId() == MuonSubdetId::CSC) {
		  CSCDetId cscId(id.rawId());
		  m_csc_station = (cscId.endcap() == 1 ? 1 : -1) * cscId.station();
		  m_csc_ring = cscId.ring();
		  m_csc_chamber = cscId.chamber();
		  m_csc_layer = cscId.layer();
		  m_csc_hitx = hitxyz.x();
		  m_csc_hity = hitxyz.y();
		  m_csc_hitz = hitxyz.z();
		  m_csc_trackx = trackxyz.x();
		  m_csc_tracky = trackxyz.y();
		  m_csc_trackz = trackxyz.z();
		  m_csc_chi2i = chi2i;
		  m_csc_survive = survive;
		  m_csc_dof = dof;
		  m_csc_chi2 = chi2;
		  m_csc->Fill();
	       }		  
	    } // end if debuggingNtuples
	 } // end loop over hits

      } // end if denom != 0 (should be two '}'s if there's a second fit)
   } // end loop over radial/non-radial

   assert(clonedHits.size() == transHits.size());
   assert(transHits.size() == TSOSes.size());
	    
   // build the trajectory
   if (clonedHits.size() > 0) {
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
   } // end if there are any clonedHits/TSOSes to work with

   // insert the trajectories into the Event
   unsigned int numTrajectories = trajectoryCollection->size();
   edm::OrphanHandle<std::vector<Trajectory> > ohTrajs = iEvent.put(trajectoryCollection);

   // also insert a trajectory <-> track association map
   std::auto_ptr<TrajTrackAssociationCollection> trajTrackMap(new TrajTrackAssociationCollection());
   if (numTrajectories == 1) {
      trajTrackMap->insert(edm::Ref<std::vector<Trajectory> >(ohTrajs, 0), edm::Ref<reco::TrackCollection>(tracks, 0));
   }
   iEvent.put(trajTrackMap);
}

// ------------ method called once each job just before starting event loop  ------------
void 
ZeroFieldMuonHIPRefitter::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
ZeroFieldMuonHIPRefitter::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(ZeroFieldMuonHIPRefitter);
