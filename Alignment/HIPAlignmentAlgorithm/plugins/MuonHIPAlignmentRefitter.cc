// -*- C++ -*-
//
// Package:    MuonHIPAlignmentRefitter
// Class:      MuonHIPAlignmentRefitter
// 
/**\class MuonHIPAlignmentRefitter MuonHIPAlignmentRefitter.cc Alignment/MuonHIPAlignmentRefitter/src/MuonHIPAlignmentRefitter.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jim Pivarski
//         Created:  Wed Dec 12 13:31:55 CST 2007
// $Id$
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "TrackingTools/TrackRefitter/interface/TrackTransformer.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "CondFormats/Alignment/interface/Definitions.h"

//
// class decleration
//

class MuonHIPAlignmentRefitter : public edm::EDProducer {
   public:
      explicit MuonHIPAlignmentRefitter(const edm::ParameterSet&);
      ~MuonHIPAlignmentRefitter();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      // ----------member data ---------------------------

      edm::InputTag m_muonSource;
      std::string m_propagatorSource;

      TrackTransformer *m_trackTransformer;
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
MuonHIPAlignmentRefitter::MuonHIPAlignmentRefitter(const edm::ParameterSet& iConfig)
{
   m_muonSource = iConfig.getParameter<edm::InputTag>("MuonSource");
   m_propagatorSource = iConfig.getParameter<std::string>("MuonPropagator");

  m_trackTransformer = new TrackTransformer(iConfig.getParameter<edm::ParameterSet>("TrackerTrackTransformer"));
  
  produces<std::vector<Trajectory> >();
  produces<TrajTrackAssociationCollection>();
}


MuonHIPAlignmentRefitter::~MuonHIPAlignmentRefitter()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
MuonHIPAlignmentRefitter::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   m_trackTransformer->setServices(iSetup);

   edm::Handle<reco::MuonCollection> muons;
   iEvent.getByLabel(m_muonSource, muons);

   edm::ESHandle<Propagator> propagator;
   iSetup.get<TrackingComponentsRecord>().get(m_propagatorSource, propagator);

   edm::ESHandle<DTGeometry> dtGeometry;
   iSetup.get<MuonGeometryRecord>().get(dtGeometry);

   edm::ESHandle<CSCGeometry> cscGeometry;
   iSetup.get<MuonGeometryRecord>().get(cscGeometry);

   edm::ESHandle<MagneticField> magneticField;
   iSetup.get<IdealMagneticFieldRecord>().get(magneticField);

   for (reco::MuonCollection::const_iterator muon = muons->begin();  muon != muons->end();  ++muon) {
      std::vector<Trajectory> trackerTrajectories = m_trackTransformer->transform(*muon->track());

      if (trackerTrajectories.size() == 1) {
	 const Trajectory trackerTrajectory = *(trackerTrajectories.begin());
	 TrajectoryStateOnSurface tracker_tsos = trackerTrajectory.lastMeasurement().forwardPredictedState();
	 TrajectoryStateOnSurface last_tsos = trackerTrajectory.lastMeasurement().forwardPredictedState();
   
	 int last_chamber = 0;
	 std::vector<double> last_x, last_xerr2, last_zforx;
	 std::vector<double> last_y, last_yerr2, last_zfory;
	 std::vector<const TrackingRecHit*> last_hits;

	 for (trackingRecHit_iterator hit = muon->standAloneMuon()->recHitsBegin();  hit != muon->standAloneMuon()->recHitsEnd();  ++hit) {
	    DetId id = (*hit)->geographicalId();

	    if (id.det() == DetId::Muon  &&  id.subdetId() != MuonSubdetId::RPC) {
	       int chamberId = -1;
	       if (id.subdetId() == MuonSubdetId::DT) {
		  DTChamberId dtChamberId(id.rawId());
		  chamberId = dtChamberId.rawId();
	       }
	       else if (id.subdetId() == MuonSubdetId::CSC) {
		  CSCDetId cscChamberId(id.rawId());
		  cscChamberId = CSCDetId(cscChamberId.endcap(), cscChamberId.station(), cscChamberId.ring(), cscChamberId.chamber(), 0);
		  chamberId = cscChamberId.rawId();
	       }
	       else assert(false);

	       // First chamber: set "last_chamber" variable
	       if (last_chamber == 0) {
		  last_chamber = chamberId;
	       }
	       // New chamber: re-orient propagator based on linear fit to chamber's 1-12 dof
	       else if (last_chamber != chamberId) {
		  GlobalPoint globalPosition = last_tsos.globalPosition();
		  GlobalVector globalDirection = last_tsos.globalDirection();

		  const Surface* surface;
		  if (DetId(last_chamber).subdetId() == MuonSubdetId::DT) {
		     surface = &(dtGeometry->idToDet(DetId(last_chamber))->surface());
		  }
		  else {
		     surface = &(cscGeometry->idToDet(DetId(last_chamber))->surface());
		  }

		  LocalPoint localPosition = surface->toLocal(globalPosition);
		  LocalVector localDirection = surface->toLocal(globalDirection);
		  // we want a normalization in which dz/dz == 1
		  localDirection /= localDirection.z();

		  if (last_zforx.size() > 3) {
		     // weighted linear fit to x(z)
		     double x_S = 0;
		     double x_Sz = 0;
		     double x_Sx = 0;
		     double x_Szz = 0;
		     double x_Szx = 0;
		     
		     for (unsigned int i = 0;  i < last_zforx.size();  i++) {
			double x = last_x[i];
			double z = last_zforx[i];
			double x_weight = 1./last_xerr2[i];

			x_S += x_weight;
			x_Sz += z * x_weight;
			x_Sx += x * x_weight;
			x_Szz += z*z * x_weight;
			x_Szx += z*x * x_weight;
		     }

		     double x_delta = x_S * x_Szz - x_Sz*x_Sz;
		     double x_intercept = (x_Szz * x_Sx - x_Sz * x_Szx) / x_delta;
//		     double x_intercept_err2 = x_Szz / x_delta;
		     double x_slope = (x_S * x_Szx - x_Sz * x_Sx) / x_delta;
//		     double x_slope_err2 = x_S / x_delta;
//		     double x_covariance = -x_Sz / x_delta;

		     double x_on_surface = x_intercept + x_slope * localPosition.z();
//		     double x_on_surface_err2 = x_intercept_err2 + localPosition.z()*localPosition.z() * x_slope_err2 + localPosition.z() * x_covariance;

		     localPosition = LocalPoint(x_on_surface, localPosition.y(), localPosition.z());
		     localDirection = LocalVector(x_slope, localDirection.y(), 1);
		  }

		  if (last_zfory.size() > 3) {
		     // weighted linear fit to y(z)
		     double y_S = 0;
		     double y_Sz = 0;
		     double y_Sy = 0;
		     double y_Szz = 0;
		     double y_Szy = 0;
		     for (unsigned int i = 0;  i < last_zfory.size();  i++) {
			double y = last_y[i];
			double z = last_zfory[i];
			double y_weight = 1./last_yerr2[i];

			y_S += y_weight;
			y_Sz += z * y_weight;
			y_Sy += y * y_weight;
			y_Szz += z*z * y_weight;
			y_Szy += z*y * y_weight;
		     }

		     double y_delta = y_S * y_Szz - y_Sz*y_Sz;
		     double y_intercept = (y_Szz * y_Sy - y_Sz * y_Szy) / y_delta;
//		     double y_intercept_err2 = y_Szz / y_delta;
		     double y_slope = (y_S * y_Szy - y_Sz * y_Sy) / y_delta;
//		     double y_slope_err2 = y_S / y_delta;
//		     double y_covariance = -y_Sz / y_delta;
		  
		     double y_on_surface = y_intercept + y_slope * localPosition.z();
//		     double y_on_surface_err2 = y_intercept_err2 + localPosition.z()*localPosition.z() * y_slope_err2 + localPosition.z() * y_covariance;

		     localPosition = LocalPoint(localPosition.x(), y_on_surface, localPosition.z());
		     localDirection = LocalVector(localDirection.x(), y_slope, 1);
		  }

		  if (last_zforx.size() > 3  ||  last_zfory.size() > 3) {
		     // make the direction normalized again
		     localDirection /= sqrt(localDirection.mag());

		     globalPosition = surface->toGlobal(localPosition);
		     globalDirection = surface->toGlobal(localDirection);
		     GlobalVector globalMomentum = globalDirection * fabs(1./last_tsos.signedInverseMomentum());

		     AlgebraicSymMatrix55 error(last_tsos.curvilinearError().matrix());
		     GlobalTrajectoryParameters globalTrajectoryParameters(globalPosition, globalMomentum, last_tsos.charge(), &*magneticField);
		     FreeTrajectoryState freeTrajectoryState(globalTrajectoryParameters, CurvilinearTrajectoryError(error));
		     last_tsos = TrajectoryStateOnSurface(freeTrajectoryState, last_tsos.surface());
		  }

		  last_x.clear();
		  last_y.clear();
		  last_zforx.clear();
		  last_zfory.clear();
		  last_xerr2.clear();
		  last_yerr2.clear();
		  last_chamber = chamberId;

		  last_hits.clear();
	       }

	       // Collect hits and propagate to surface
	       LocalPoint chamberPoint;
	       align::RotationType layerRot, chamberRot;

	       last_hits.push_back(&**hit);
	       GlobalPoint global_hit_position;
	       TrajectoryStateOnSurface extrapolation, fromtracker;
	       if (id.subdetId() == MuonSubdetId::DT) {
		  extrapolation = propagator->propagate(last_tsos, dtGeometry->idToDet(id)->surface());
		  fromtracker = propagator->propagate(tracker_tsos, dtGeometry->idToDet(id)->surface());
		  LocalPoint local = (*hit)->localPosition();
		  if (extrapolation.isValid()) {
		     // The extrapolated track is used to set a y position for the hit;
		     // y positions are always small corrections to chamber coordinates,
		     // applicable only when layers are rotated inside the chamber
		     local = LocalPoint(local.x(), extrapolation.localPosition().y(), 0);
		  }

		  GlobalPoint globalPoint = dtGeometry->idToDet(id)->surface().toGlobal(local);
		  chamberPoint = dtGeometry->idToDet(chamberId)->surface().toLocal(globalPoint);
		  layerRot = dtGeometry->idToDet(id)->surface().rotation();
		  chamberRot = dtGeometry->idToDet(chamberId)->surface().rotation();
		  global_hit_position = globalPoint;
	       }
	       else if (id.subdetId() == MuonSubdetId::CSC) {
		  extrapolation = propagator->propagate(last_tsos, cscGeometry->idToDet(id)->surface());
		  fromtracker = propagator->propagate(tracker_tsos, cscGeometry->idToDet(id)->surface());
		  GlobalPoint globalPoint = cscGeometry->idToDet(id)->surface().toGlobal((*hit)->localPosition());
		  chamberPoint = cscGeometry->idToDet(chamberId)->surface().toLocal(globalPoint);
		  layerRot = cscGeometry->idToDet(id)->surface().rotation();
		  chamberRot = cscGeometry->idToDet(chamberId)->surface().rotation();
		  global_hit_position = globalPoint;
	       }
	       else assert(false);

	       double lRxx = layerRot.xx();
	       double lRxy = layerRot.xy();
	       double lRxz = layerRot.xz();
	       double lRyx = layerRot.yx();
	       double lRyy = layerRot.yy();
	       double lRyz = layerRot.yz();
//	       double lRzx = layerRot.zx();
//	       double lRzy = layerRot.zy();
//	       double lRzz = layerRot.zz();
	       double cRxx = chamberRot.xx();
	       double cRxy = chamberRot.xy();
	       double cRxz = chamberRot.xz();
	       double cRyx = chamberRot.yx();
	       double cRyy = chamberRot.yy();
	       double cRyz = chamberRot.yz();
//	       double cRzx = chamberRot.zx();
//	       double cRzy = chamberRot.zy();
//	       double cRzz = chamberRot.zz();
	       
	       if (id.subdetId() == MuonSubdetId::DT) {
		  if (fabs(cRxx*lRxx + cRxy*lRxy + cRxz*lRxz) > fabs(cRxx*lRyx + cRxy*lRyy + cRxz*lRyz)) {
		     last_x.push_back(chamberPoint.x());
		     last_xerr2.push_back(pow((cRxx*lRxx + cRxy*lRxy + cRxz*lRxz), 2) * (*hit)->localPositionError().xx() +
					  pow((cRxx*lRyx + cRxy*lRyy + cRxz*lRyz), 2) * (*hit)->localPositionError().yy());
		     last_zforx.push_back(chamberPoint.z());
		  }
		  else {
		     last_y.push_back(chamberPoint.y());
		     last_yerr2.push_back(pow((cRyx*lRxx + cRyy*lRxy + cRyz*lRxz), 2) * (*hit)->localPositionError().xx() +
					  pow((cRyx*lRyx + cRyy*lRyy + cRyz*lRyz), 2) * (*hit)->localPositionError().yy());
		     last_zfory.push_back(chamberPoint.z());
		  }
	       }
	       else if (id.subdetId() == MuonSubdetId::CSC) {
		  last_x.push_back(chamberPoint.x());
		  last_xerr2.push_back(pow((cRxx*lRxx + cRxy*lRxy + cRxz*lRxz), 2) * (*hit)->localPositionError().xx() +
				       pow((cRxx*lRyx + cRxy*lRyy + cRxz*lRyz), 2) * (*hit)->localPositionError().yy());
		  last_zforx.push_back(chamberPoint.z());

		  last_y.push_back(chamberPoint.y());
		  last_yerr2.push_back(pow((cRyx*lRxx + cRyy*lRxy + cRyz*lRxz), 2) * (*hit)->localPositionError().xx() +
				       pow((cRyx*lRyx + cRyy*lRyy + cRyz*lRyz), 2) * (*hit)->localPositionError().yy());
		  last_zfory.push_back(chamberPoint.z());
	       }
	       else assert(false);

	       if (extrapolation.isValid()) {
		  last_tsos = extrapolation;
	       }
	    } // end if Muon and not RPC
	 } // end loop over standAlone hits
      } // end if successfully refit tracker track
   } // end loop over muons

   std::auto_ptr<std::vector<Trajectory> > trajectoryCollection(new std::vector<Trajectory>);
   std::auto_ptr<TrajTrackAssociationCollection> trajTrackMap(new TrajTrackAssociationCollection);
   iEvent.put(trajectoryCollection);
   iEvent.put(trajTrackMap);
}

// ------------ method called once each job just before starting event loop  ------------
void 
MuonHIPAlignmentRefitter::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
MuonHIPAlignmentRefitter::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(MuonHIPAlignmentRefitter);
