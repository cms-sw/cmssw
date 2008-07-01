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
// $Id: ZeroFieldMuonHIPRefitter.cc,v 1.10 2008/06/16 15:43:14 pivarski Exp $
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
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"
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
#include "TrackingTools/TrackRefitter/interface/TrackTransformer.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"
#include "TH1F.h"

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
      TrackTransformer *m_trackTransformer;

      edm::InputTag m_input;
      int m_minDOF;
      std::string m_propagator;
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
   , m_minDOF(iConfig.getParameter<int>("minDOF"))
   , m_propagator(iConfig.getParameter<std::string>("propagator"))
{
   m_trackTransformer = new TrackTransformer(iConfig.getParameter<edm::ParameterSet>("TrackerTrackTransformer"));

   produces<std::vector<Trajectory> >();
   produces<TrajTrackAssociationCollection>();
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
   m_trackTransformer->setServices(iSetup);

   edm::ESHandle<Propagator> propagator;
   iSetup.get<TrackingComponentsRecord>().get(m_propagator, propagator);

   edm::Handle<reco::TrackCollection> tracks;
   iEvent.getByLabel(m_input, tracks);

   edm::ESHandle<TrackerGeometry> trackerGeometry;
   iSetup.get<TrackerDigiGeometryRecord>().get(trackerGeometry);

   edm::ESHandle<DTGeometry> dtGeometry;
   iSetup.get<MuonGeometryRecord>().get(dtGeometry);

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

//    GlobalPoint cheatPoint(0.52, -0.84, 1.22);
//    GlobalVector cheatVector(-8.4, 15.2, -9.6);
//    std::vector<double> cheatx, cheaty, cheatz;
//    for (double z = 0.;  z < 100.;  z += 1.) {
//       GlobalPoint cheatHit = cheatPoint + cheatVector * z;
//       cheatx.push_back(cheatHit.x());
//       cheaty.push_back(cheatHit.y());
//       cheatz.push_back(cheatHit.z());
//    }

   // we want the independent variable of the fit to be roughly collinear with the resulting line
   // (so the final fitted slope will be a small correction)
   // we convert from local to global, and then left-multiply by this rotation
   math::XYZVector direction = best_track->momentum();
//   math::XYZVector direction = math::XYZVector(cheatVector.x(), cheatVector.y(), cheatVector.z());
// CHEAT

   double theta1 = atan2(-direction.x(), direction.y());
   double theta2 = atan2(direction.z(), sqrt(direction.perp2()));
   align::RotationType coordrot(             cos(theta1),  sin(theta1),             0.,
                                 sin(theta1)*sin(theta2), -cos(theta1)*sin(theta2), cos(theta2),
                                -sin(theta1)*cos(theta2),  cos(theta1)*cos(theta2), sin(theta2));
   align::RotationType coordrotinv(cos(theta1),  sin(theta1)*sin(theta2), -sin(theta1)*cos(theta2),
                                   sin(theta1), -cos(theta1)*sin(theta2),  cos(theta1)*cos(theta2),
                                            0.,              cos(theta2),              sin(theta2));

//    align::RotationType coordrot(1., 0., 0.,
// 				0., 1., 0.,
// 				0., 0., 1.);
//    align::RotationType coordrotinv(1., 0., 0.,
// 				   0., 1., 0.,
// 				   0., 0., 1.);

   // these x and y values are perpendicular to the direction-of-momentum axis
   std::vector<const TrackingRecHit*> hits;
   std::vector<double> listx, listy, listz, listXX, listXY, listYY;
   double SXX, SxXX, SxXY, SXY, SxzXX, SxzXY, SyXY, SYY, SyYY, SyzXY, SyzYY, SzXX, SzXY, SzYY, SzzXX, SzzXY, SzzYY;
   SXX = SxXX = SxXY = SXY = SxzXX = SxzXY = SyXY = SYY = SyYY = SyzXY = SyzYY = SzXX = SzXY = SzYY = SzzXX = SzzXY = SzzYY = 0.;

//   int cheati = 0;
   for (trackingRecHit_iterator hit = best_track->recHitsBegin();  hit != best_track->recHitsEnd();  ++hit) {
      DetId id = (*hit)->geographicalId();
      if (id.det() == DetId::Tracker) {
	 LocalPoint localPoint = (*hit)->localPosition();
	 LocalError localError = (*hit)->localPositionError();

	 const Surface& surface = trackerGeometry->idToDet(id)->surface();
	 
	 GlobalPoint position = surface.toGlobal(localPoint);
//	 GlobalPoint position(cheatx[cheati], cheaty[cheati], cheatz[cheati]);
//	 cheati++;
// CHEAT

	 position = GlobalPoint(coordrot.xx()*position.x() + coordrot.xy()*position.y() + coordrot.xz()*position.z(),
				coordrot.yx()*position.x() + coordrot.yy()*position.y() + coordrot.yz()*position.z(),
				coordrot.zx()*position.x() + coordrot.zy()*position.y() + coordrot.zz()*position.z());
	 
//	 std::cout << "before = " << std::endl << error << std::endl;

// 	 GlobalVector vect = surface.toGlobal(LocalVector(localPoint.x(), localPoint.y(), localPoint.z()));
// 	 vect = GlobalVector(coordrot.xx()*vect.x() + coordrot.xy()*vect.y() + coordrot.xz()*vect.z(),
// 			     coordrot.yx()*vect.x() + coordrot.yy()*vect.y() + coordrot.yz()*vect.z(),
// 			     coordrot.zx()*vect.x() + coordrot.zy()*vect.y() + coordrot.zz()*vect.z());

 	 align::RotationType rotation = coordrot * surface.rotation().transposed();

// 	 GlobalPoint vect2(rotation.xx()*localPoint.x() + rotation.xy()*localPoint.y() + rotation.xz()*localPoint.z(),
// 			   rotation.yx()*localPoint.x() + rotation.yy()*localPoint.y() + rotation.yz()*localPoint.z(),
// 			   rotation.zx()*localPoint.x() + rotation.zy()*localPoint.y() + rotation.zz()*localPoint.z());

// 	 std::cout << "vect = " << vect << " " << vect2 << std::endl;
// 	 // this works.  what's wrong with my error propagation?

	 
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



// 	 if ((*hit)->dimension() == 1) {
// 	    error[0][0] = 1./localError.xx();
// 	    error[0][1] = 0.;
// 	    error[1][1] = 0.;
// 	 }
// 	 else {
// 	    error[0][0] = localError.xx();
// 	    error[0][1] = localError.xy();
// 	    error[1][1] = localError.yy();
// 	 }

//	 std::cout << "after = " << std::endl << error << std::endl;

	 // these x and y values are perpendicular to the direction-of-momentum axis
	 double xi = position.x();
	 double yi = position.y();
	 double zi = position.z();
	 double XX = error[0][0];
	 double XY = error[0][1];
	 double YY = error[1][1];

	 hits.push_back(&(**hit));
	 listx.push_back(xi);
	 listy.push_back(yi);
	 listz.push_back(zi);
	 listXX.push_back(XX);
	 listXY.push_back(XY);
	 listYY.push_back(YY);

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

      } // end if tracker hit
   } // end loop over hits

   // calculate the least-squares fit
   double denom = (SzzXX*(SXX*(SzzYY*SYY - pow(SzYY,2)) - pow(SzXY,2)*SYY - SzzYY*pow(SXY,2) + 2*SzXY*SzYY*SXY) + SzzXY*(SzXY*(2*SzXX*SYY + 2*SzYY*SXX) - 2*SzXX*SzYY*SXY - 2*pow(SzXY,2)*SXY) + pow(SzzXY,2)*(pow(SXY,2) - SXX*SYY) + pow(SzXX,2)*(pow(SzYY,2) - SzzYY*SYY) + 2*SzXX*SzXY*SzzYY*SXY + pow(SzXY,2)*(-SzzYY*SXX - 2*SzXX*SzYY) + pow(SzXY,4));
   double a = (-SzXX*(SyXY*(SzzYY*SYY - pow(SzYY,2)) + SxXX*(SzzYY*SYY - pow(SzYY,2)) + ((-SyYY - SxXY)*SzzYY + (SyzYY + SxzXY)*SzYY)*SXY) - SyzXY*(SXX*(pow(SzYY,2) - SzzYY*SYY) + SzzYY*pow(SXY,2)) - SxzXX*(SXX*(pow(SzYY,2) - SzzYY*SYY) + SzzYY*pow(SXY,2)) - SzzXY*(SXX*(SyzYY*SYY + SxzXY*SYY + (-SyYY - SxXY)*SzYY) + SzXY*(-SyXY*SYY - SxXX*SYY + (SyYY + SxXY)*SXY) + (-SyzYY - SxzXY)*pow(SXY,2) + (SyXY*SzYY + SxXX*SzYY)*SXY) - SzXY*(SzXX*(-SyzYY*SYY - SxzXY*SYY + (SyYY + SxXY)*SzYY) + (-SyXY*SzzYY - SxXX*SzzYY)*SXY - 2*SyzXY*SzYY*SXY - 2*SxzXX*SzYY*SXY + ((SyYY + SxXY)*SzzYY + (-SyzYY - SxzXY)*SzYY)*SXX) - pow(SzXY,2)*(SyzXY*SYY + SxzXX*SYY + (SyzYY + SxzXY)*SXY + SyXY*SzYY + SxXX*SzYY) - (-SyYY - SxXY)*pow(SzXY,3))/denom;
   double b = (SzzXX*(SyXY*(SzzYY*SYY - pow(SzYY,2)) + SxXX*(SzzYY*SYY - pow(SzYY,2)) + SzXY*(-SyzYY*SYY - SxzXY*SYY + (SyYY + SxXY)*SzYY) + ((-SyYY - SxXY)*SzzYY + (SyzYY + SxzXY)*SzYY)*SXY) + SzXX*(SyzXY*(pow(SzYY,2) - SzzYY*SYY) + SxzXX*(pow(SzYY,2) - SzzYY*SYY)) + SzzXY*(SzXX*(SyzYY*SYY + SxzXY*SYY + (-SyYY - SxXY)*SzYY) + SzXY*(SyzXY*SYY + SxzXX*SYY + (-SyzYY - SxzXY)*SXY + 2*SyXY*SzYY + 2*SxXX*SzYY) - SyzXY*SzYY*SXY - SxzXX*SzYY*SXY + (-SyYY - SxXY)*pow(SzXY,2)) + pow(SzzXY,2)*(-SyXY*SYY - SxXX*SYY + (SyYY + SxXY)*SXY) + SzXY*(SyzXY*SzzYY*SXY + SxzXX*SzzYY*SXY + SzXX*((SyYY + SxXY)*SzzYY + (-SyzYY - SxzXY)*SzYY)) + pow(SzXY,2)*(-SyXY*SzzYY - SxXX*SzzYY - SyzXY*SzYY - SxzXX*SzYY) + (SyzYY + SxzXY)*pow(SzXY,3))/denom;
   double c = (-SzzXY*(SyzXY*(SXX*SYY - pow(SXY,2)) + SxzXX*(SXX*SYY - pow(SXY,2)) + SzXX*(-SyXY*SYY - SxXX*SYY + (SyYY + SxXY)*SXY) + SzXY*((SyXY + SxXX)*SXY + (-SyYY - SxXY)*SXX)) - SzzXX*(SXX*(-SyzYY*SYY - SxzXY*SYY + (SyYY + SxXY)*SzYY) + SzXY*(SyXY*SYY + SxXX*SYY + (-SyYY - SxXY)*SXY) + (SyzYY + SxzXY)*pow(SXY,2) + (-SyXY*SzYY - SxXX*SzYY)*SXY) - SzXY*(SzXX*(-SyzXY*SYY - SxzXX*SYY + (-2*SyzYY - 2*SxzXY)*SXY + SyXY*SzYY + SxXX*SzYY) - SyzXY*SzYY*SXX - SxzXX*SzYY*SXX) - pow(SzXX,2)*(SyzYY*SYY + SxzXY*SYY + (-SyYY - SxXY)*SzYY) - SzXX*(SyzXY*SzYY*SXY + SxzXX*SzYY*SXY) - pow(SzXY,2)*(SyzXY*SXY + SxzXX*SXY + (SyzYY + SxzXY)*SXX + (SyYY + SxXY)*SzXX) - (-SyXY - SxXX)*pow(SzXY,3))/denom;
   double d = (SzzXX*(SzXY*((SyzYY + SxzXY)*SXY + SyXY*SzYY + SxXX*SzYY) + (-SyXY*SzzYY - SxXX*SzzYY)*SXY + ((SyYY + SxXY)*SzzYY + (-SyzYY - SxzXY)*SzYY)*SXX + (-SyYY - SxXY)*pow(SzXY,2)) + SzzXY*(SzXX*((-SyzYY - SxzXY)*SXY - SyXY*SzYY - SxXX*SzYY) + SzXY*(-SyzXY*SXY - SxzXX*SXY + (SyzYY + SxzXY)*SXX + (2*SyYY + 2*SxXY)*SzXX) + SyzXY*SzYY*SXX + SxzXX*SzYY*SXX + (-SyXY - SxXX)*pow(SzXY,2)) + SzXX*(SyzXY*SzzYY*SXY + SxzXX*SzzYY*SXY) + pow(SzzXY,2)*((SyXY + SxXX)*SXY + (-SyYY - SxXY)*SXX) + SzXY*(-SyzXY*SzzYY*SXX - SxzXX*SzzYY*SXX + SzXX*(SyXY*SzzYY + SxXX*SzzYY - SyzXY*SzYY - SxzXX*SzYY)) + pow(SzXX,2)*((-SyYY - SxXY)*SzzYY + (SyzYY + SxzXY)*SzYY) + (SyzXY + SxzXX)*pow(SzXY,3) + (-SyzYY - SxzXY)*SzXX*pow(SzXY,2))/denom;

   GlobalVector fitslope = GlobalVector(a, c, 1.) / sqrt(pow(a,2) + pow(c,2) + 1.);
   GlobalPoint fitpoint(b, d, 0.);

   // left-multiply by the inverse rotation
   fitslope = GlobalVector(coordrot.xx() * fitslope.x() + coordrot.yx() * fitslope.y() + coordrot.zx() * fitslope.z(),
			   coordrot.xy() * fitslope.x() + coordrot.yy() * fitslope.y() + coordrot.zy() * fitslope.z(),
			   coordrot.xz() * fitslope.x() + coordrot.yz() * fitslope.y() + coordrot.zz() * fitslope.z());
   fitpoint = GlobalPoint(coordrot.xx() * fitpoint.x() + coordrot.yx() * fitpoint.y() + coordrot.zx() * fitpoint.z(),
			  coordrot.xy() * fitpoint.x() + coordrot.yy() * fitpoint.y() + coordrot.zy() * fitpoint.z(),
			  coordrot.xz() * fitpoint.x() + coordrot.yz() * fitpoint.y() + coordrot.zz() * fitpoint.z());

   double chi2 = 0.;
   int dof = 0;
   std::vector<const TrackingRecHit*>::const_iterator hit = hits.begin();
   std::vector<double>::const_iterator xi = listx.begin();
   std::vector<double>::const_iterator yi = listy.begin();
   std::vector<double>::const_iterator zi = listz.begin();
   std::vector<double>::const_iterator XX = listXX.begin();
   std::vector<double>::const_iterator XY = listXY.begin();
   std::vector<double>::const_iterator YY = listYY.begin();

//   cheati = 0;
   for (;  hit != hits.end();  ++hit, ++xi, ++yi, ++zi, ++XX, ++XY, ++YY) {
      GlobalPoint realspace = GlobalPoint(b, d, 0.) + GlobalVector(a, c, 1.) * (*zi);
      realspace = GlobalPoint(coordrot.xx() * realspace.x() + coordrot.yx() * realspace.y() + coordrot.zx() * realspace.z(),
			      coordrot.xy() * realspace.x() + coordrot.yy() * realspace.y() + coordrot.zy() * realspace.z(),
			      coordrot.xz() * realspace.x() + coordrot.yz() * realspace.y() + coordrot.zz() * realspace.z());

      GlobalPoint realhit = trackerGeometry->idToDet((*hit)->geographicalId())->toGlobal((*hit)->localPosition());
//      GlobalPoint realhit(cheatx[cheati], cheaty[cheati], cheatz[cheati]);
//      cheati++;
// CHEAT      

      double x = a * (*zi) + b;
      double y = c * (*zi) + d;

      double chi2i = (x - (*xi))*(x - (*xi))*(*XX) + 2*(x - (*xi))*(y - (*yi))*(*XY) + (y - (*yi))*(y - (*yi))*(*YY);
      chi2 += chi2i;
      if ((*hit)->dimension() == 1) dof += 1;
      else dof += 2;

      std::cout << "x y " << x << " " << y << " xresid " << (x - (*xi)) << " +- " << sqrt(1./(*XX)) << " yresid " << (y - (*yi)) << " +- " << sqrt(1./(*YY)) << " xyresid " << (2*(x - (*xi))*(y - (*yi))) << " +- " << (1./(*XY)) << " chi2i = " << chi2i << std::endl;

   }
   dof -= 4;

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
