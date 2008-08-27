// -*- C++ -*-
//
// Package:    ZeroFieldStandAloneMuonHIPRefitter
// Class:      ZeroFieldStandAloneMuonHIPRefitter
// 
/**\class ZeroFieldStandAloneMuonHIPRefitter ZeroFieldStandAloneMuonHIPRefitter.cc Alignment/ZeroFieldStandAloneMuonHIPRefitter/src/ZeroFieldStandAloneMuonHIPRefitter.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jim Pivarski
//         Created:  Thu Jul 17 00:02:16 CEST 2008
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

class ZeroFieldStandAloneMuonHIPRefitter : public edm::EDProducer {
   public:
      explicit ZeroFieldStandAloneMuonHIPRefitter(const edm::ParameterSet&);
      ~ZeroFieldStandAloneMuonHIPRefitter();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      // ----------member data ---------------------------
      edm::InputTag m_input;
      bool m_isDT;
      int m_minXmeasuring, m_minZmeasuring;
      std::vector<int> m_mustContainCSC;

      TH1F *m_redchi2_10;
      TH1F *m_redchi2_100;
      TH1F *m_redchi2_1000;
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
ZeroFieldStandAloneMuonHIPRefitter::ZeroFieldStandAloneMuonHIPRefitter(const edm::ParameterSet& iConfig)
   : m_input(iConfig.getParameter<edm::InputTag>("input"))
   , m_isDT(iConfig.getParameter<bool>("isDT"))
   , m_minXmeasuring(iConfig.getUntrackedParameter<int>("minXmeasuring", 4))
   , m_minZmeasuring(iConfig.getUntrackedParameter<int>("minZmeasuring", 4))
   , m_mustContainCSC(iConfig.getUntrackedParameter<std::vector<int> >("mustContainCSC", std::vector<int>()))
{
   produces<std::vector<Trajectory> >();
   produces<TrajTrackAssociationCollection>();

   edm::Service<TFileService> tfile;
   m_redchi2_10 = tfile->make<TH1F>("redchi2_10", "redchi2_10", 100, 0., 10.);
   m_redchi2_100 = tfile->make<TH1F>("redchi2_100", "redchi2_100", 100, 0., 100.);
   m_redchi2_1000 = tfile->make<TH1F>("redchi2_1000", "redchi2_1000", 100, 0., 1000.);
}


ZeroFieldStandAloneMuonHIPRefitter::~ZeroFieldStandAloneMuonHIPRefitter()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
ZeroFieldStandAloneMuonHIPRefitter::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   edm::Handle<reco::TrackCollection> tracks;
   iEvent.getByLabel(m_input, tracks);

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

   // Remember which trajectory is associated with which track
   std::map<edm::Ref<std::vector<Trajectory> >::key_type, edm::Ref<reco::TrackCollection>::key_type> reference_map;
   edm::Ref<std::vector<Trajectory> >::key_type trajCounter = 0;
   edm::Ref<reco::TrackCollection>::key_type trackCounter = 0;

   for (reco::TrackCollection::const_iterator track = tracks->begin();  track != tracks->end();  ++track) {
      // independent variable for the fit
      math::XYZVector direction(0., 0., 1.);
      
      // if DT, independent variable needs to be perpendicular to most of the stations involved in the fit
      // also, make sure there are enough x-measuring hits and enough z-measuring hits
      if (m_isDT) {
	 int xmeasuring = 0;
	 int zmeasuring = 0;

	 int sectorsHist[12];
	 for (int i = 0;  i < 12;  i++) sectorsHist[i] = 0;
	 for (trackingRecHit_iterator hit = track->recHitsBegin();  hit != track->recHitsEnd();  ++hit) {
	    DetId id = (*hit)->geographicalId();
	    if (id.det() == DetId::Muon  &&  id.subdetId() == MuonSubdetId::DT) {
	       DTLayerId dtId(id.rawId());
	       if (dtId.station() != 4) {
		  sectorsHist[dtId.sector()]++;
	       }

	       if (dtId.superlayer() == 2) zmeasuring++;
	       else xmeasuring++;
	    }
	 }
	 
	 if (xmeasuring < m_minXmeasuring  ||  zmeasuring < m_minZmeasuring) continue;

	 int maxSector = -1;
	 int sectorsMax = 0;
	 for (int i = 0;  i < 12;  i++) {
	    if (sectorsHist[i] > sectorsMax) {
	       maxSector = i;
	       sectorsMax = sectorsHist[i];
	    }
	 }

	 double average_phi = 0;
	 double N_phi = 0;
	 for (trackingRecHit_iterator hit = track->recHitsBegin();  hit != track->recHitsEnd();  ++hit) {
	    DetId id = (*hit)->geographicalId();
	    if (id.det() == DetId::Muon  &&  id.subdetId() == MuonSubdetId::DT) {
	       DTChamberId dtId(id.rawId());
	       if (dtId.station() != 4) {
		  GlobalVector chamberNorm = dtGeometry->idToDet(dtId)->toGlobal(LocalVector(0., 0., 1.));
		  double phi = atan2(chamberNorm.y(), chamberNorm.x());
		  average_phi += phi;
		  N_phi += 1.;
	       }
	    }
	 }
	 if (N_phi == 0) continue;
	 average_phi /= N_phi;

	 direction = math::XYZVector(cos(average_phi), sin(average_phi), 0.);
      }

      // get a coordinate rotation such that the transformed points' "z" is the independent variable
      double theta1 = atan2(-direction.x(), direction.y());
      double theta2 = atan2(direction.z(), sqrt(direction.perp2()));
      align::RotationType coordrot(             cos(theta1),  sin(theta1),             0.,
                                    sin(theta1)*sin(theta2), -cos(theta1)*sin(theta2), cos(theta2),
                                   -sin(theta1)*cos(theta2),  cos(theta1)*cos(theta2), sin(theta2));

      edm::OwnVector<TrackingRecHit> clonedHits;
      std::vector<TrajectoryMeasurement::ConstRecHitPointer> transHits;
      std::vector<TrajectoryStateOnSurface> TSOSes;

      std::vector<const TrackingRecHit*> hits;
      std::vector<double> listx, listy, listz, listXX, listXY, listYY;
      double SXX, SxXX, SxXY, SXY, SxzXX, SxzXY, SyXY, SYY, SyYY, SyzXY, SyzYY, SzXX, SzXY, SzYY, SzzXX, SzzXY, SzzYY;
      SXX = SxXX = SxXY = SXY = SxzXX = SxzXY = SyXY = SYY = SyYY = SyzXY = SyzYY = SzXX = SzXY = SzYY = SzzXX = SzzXY = SzzYY = 0.;

      bool contains_ME11 = false;
      bool contains_ME12 = false;
      bool contains_ME13 = false;
      bool contains_ME14 = false;
      bool contains_ME21 = false;
      bool contains_ME22 = false;
      bool contains_ME31 = false;
      bool contains_ME32 = false;
      bool contains_ME41 = false;

      for (trackingRecHit_iterator hit = track->recHitsBegin();  hit != track->recHitsEnd();  ++hit) {
	 DetId id = (*hit)->geographicalId();

	 const Surface *layerSurface = NULL;
	 if (id.det() == DetId::Muon  &&  id.subdetId() == MuonSubdetId::DT) {
	    layerSurface = &(dtGeometry->idToDet(id)->surface());
	 }
	 else if (id.det() == DetId::Muon  &&  id.subdetId() == MuonSubdetId::CSC) {
	    layerSurface = &(cscGeometry->idToDet(id)->surface());

	    CSCDetId cscid(id.rawId());
	    if (cscid.station() == 1  &&  cscid.ring() == 1) contains_ME11 = true;
	    if (cscid.station() == 1  &&  cscid.ring() == 2) contains_ME12 = true;
	    if (cscid.station() == 1  &&  cscid.ring() == 3) contains_ME13 = true;
	    if (cscid.station() == 1  &&  cscid.ring() == 4) contains_ME14 = true;
	    if (cscid.station() == 2  &&  cscid.ring() == 1) contains_ME21 = true;
	    if (cscid.station() == 2  &&  cscid.ring() == 2) contains_ME22 = true;
	    if (cscid.station() == 3  &&  cscid.ring() == 1) contains_ME31 = true;
	    if (cscid.station() == 3  &&  cscid.ring() == 2) contains_ME32 = true;
	    if (cscid.station() == 4  &&  cscid.ring() == 1) contains_ME41 = true;
	 }

	 if (layerSurface != NULL) {
	    TrajectoryMeasurement::ConstRecHitPointer hitPtr(muonTransBuilder.build(&**hit, globalGeometry));

	    LocalPoint localPoint = (*hit)->localPosition();
	    AlgebraicSymMatrix localErrorWithAPE = hitPtr->parametersError();
	    double sigma_xx = localErrorWithAPE[0][0];
	    double sigma_xy = (localErrorWithAPE.num_row() == 1 ? 0. : localErrorWithAPE[0][1]);
	    double sigma_yy = (localErrorWithAPE.num_row() == 1 ? 0. : localErrorWithAPE[1][1]);

	    if ((*hit)->dimension() == 1.) {
	       localPoint = LocalPoint(localPoint.x(), 0., 0.);
	       sigma_xy = 0.;
	       sigma_yy = 1e6;
	    }

	    GlobalPoint globalPoint = layerSurface->toGlobal(localPoint);
	    align::RotationType rotation, error;

	    // point in the direction of fitting
	    globalPoint = GlobalPoint(coordrot.xx()*globalPoint.x() + coordrot.xy()*globalPoint.y() + coordrot.xz()*globalPoint.z(),
				      coordrot.yx()*globalPoint.x() + coordrot.yy()*globalPoint.y() + coordrot.yz()*globalPoint.z(),
				      coordrot.zx()*globalPoint.x() + coordrot.zy()*globalPoint.y() + coordrot.zz()*globalPoint.z());
	    rotation = coordrot * layerSurface->rotation().transposed();

	    error = rotation * align::RotationType(sigma_xx, sigma_xy, 0, sigma_xy, sigma_yy, 0, 0, 0, 0) * rotation.transposed();

	    AlgebraicSymMatrix globalError(2);
	    globalError[0][0] = error.xx();
	    globalError[1][1] = error.yy();
	    globalError[0][1] = error.yx();

	    double xi = globalPoint.x();
	    double yi = globalPoint.y();
	    double zi = globalPoint.z();

	    // explicitly drop the CSCs if in a rotated system, since their transfomed error matrices might not be invertable
	    bool include_in_fit = true;
	    if (m_isDT) include_in_fit = (id.det() == DetId::Muon  &&  id.subdetId() == MuonSubdetId::DT);
	    else include_in_fit = (id.det() == DetId::Muon  &&  id.subdetId() == MuonSubdetId::CSC);

	    int ierr;
	    globalError.invert(ierr);
	    if (ierr != 0  &&  include_in_fit) {
	       edm::LogError("ZeroFieldStandAloneMuonHIPRefitter") << "Matrix inversion failed!  ierr = " << ierr << " subdetid = " << id.subdetId() << " matrix = " << std::endl << globalError << std::endl;
	       return;
	    }
	    double XX = globalError[0][0];
	    double XY = globalError[0][1];
	    double YY = globalError[1][1];

	    // push everything into the vectors to keep them all synchronized, regardless of whether they're included in the fit
	    transHits.push_back(hitPtr);
	    hits.push_back(&**hit);
	    listx.push_back(xi);
	    listy.push_back(yi);
	    listz.push_back(zi);
	    listXX.push_back(XX);
	    listXY.push_back(XY);
	    listYY.push_back(YY);

	    // only when you increment these counters are points really used in the fit
	    if (include_in_fit) {
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
	    }

	 } // end if CSC	 
      } // end loop over hits

      for (std::vector<int>::const_iterator must = m_mustContainCSC.begin();  must != m_mustContainCSC.end();  ++must) {
	 if (*must == 11) {
	    if (!contains_ME11) return;
	 }
	 else if (*must == 12) {
	    if (!contains_ME12) return;
	 }
	 else if (*must == 13) {
	    if (!contains_ME13) return;
	 }
	 else if (*must == 14) {
	    if (!contains_ME14) return;
	 }
	 else if (*must == 21) {
	    if (!contains_ME21) return;
	 }
	 else if (*must == 22) {
	    if (!contains_ME22) return;
	 }
	 else if (*must == 31) {
	    if (!contains_ME31) return;
	 }
	 else if (*must == 32) {
	    if (!contains_ME32) return;
	 }
	 else if (*must == 41) {
	    if (!contains_ME41) return;
	 }
	 else assert(false);  // really ought to be a ConfigError exception
      }

      // calculate the least-squares fit
      double denom = (SzzXX*(SXX*(SzzYY*SYY - pow(SzYY,2)) - pow(SzXY,2)*SYY - SzzYY*pow(SXY,2) + 2*SzXY*SzYY*SXY) + SzzXY*(SzXY*(2*SzXX*SYY + 2*SzYY*SXX) - 2*SzXX*SzYY*SXY - 2*pow(SzXY,2)*SXY) + pow(SzzXY,2)*(pow(SXY,2) - SXX*SYY) + pow(SzXX,2)*(pow(SzYY,2) - SzzYY*SYY) + 2*SzXX*SzXY*SzzYY*SXY + pow(SzXY,2)*(-SzzYY*SXX - 2*SzXX*SzYY) + pow(SzXY,4));
      double a = (-SzXX*(SyXY*(SzzYY*SYY - pow(SzYY,2)) + SxXX*(SzzYY*SYY - pow(SzYY,2)) + ((-SyYY - SxXY)*SzzYY + (SyzYY + SxzXY)*SzYY)*SXY) - SyzXY*(SXX*(pow(SzYY,2) - SzzYY*SYY) + SzzYY*pow(SXY,2)) - SxzXX*(SXX*(pow(SzYY,2) - SzzYY*SYY) + SzzYY*pow(SXY,2)) - SzzXY*(SXX*(SyzYY*SYY + SxzXY*SYY + (-SyYY - SxXY)*SzYY) + SzXY*(-SyXY*SYY - SxXX*SYY + (SyYY + SxXY)*SXY) + (-SyzYY - SxzXY)*pow(SXY,2) + (SyXY*SzYY + SxXX*SzYY)*SXY) - SzXY*(SzXX*(-SyzYY*SYY - SxzXY*SYY + (SyYY + SxXY)*SzYY) + (-SyXY*SzzYY - SxXX*SzzYY)*SXY - 2*SyzXY*SzYY*SXY - 2*SxzXX*SzYY*SXY + ((SyYY + SxXY)*SzzYY + (-SyzYY - SxzXY)*SzYY)*SXX) - pow(SzXY,2)*(SyzXY*SYY + SxzXX*SYY + (SyzYY + SxzXY)*SXY + SyXY*SzYY + SxXX*SzYY) - (-SyYY - SxXY)*pow(SzXY,3))/denom;
      double b = (SzzXX*(SyXY*(SzzYY*SYY - pow(SzYY,2)) + SxXX*(SzzYY*SYY - pow(SzYY,2)) + SzXY*(-SyzYY*SYY - SxzXY*SYY + (SyYY + SxXY)*SzYY) + ((-SyYY - SxXY)*SzzYY + (SyzYY + SxzXY)*SzYY)*SXY) + SzXX*(SyzXY*(pow(SzYY,2) - SzzYY*SYY) + SxzXX*(pow(SzYY,2) - SzzYY*SYY)) + SzzXY*(SzXX*(SyzYY*SYY + SxzXY*SYY + (-SyYY - SxXY)*SzYY) + SzXY*(SyzXY*SYY + SxzXX*SYY + (-SyzYY - SxzXY)*SXY + 2*SyXY*SzYY + 2*SxXX*SzYY) - SyzXY*SzYY*SXY - SxzXX*SzYY*SXY + (-SyYY - SxXY)*pow(SzXY,2)) + pow(SzzXY,2)*(-SyXY*SYY - SxXX*SYY + (SyYY + SxXY)*SXY) + SzXY*(SyzXY*SzzYY*SXY + SxzXX*SzzYY*SXY + SzXX*((SyYY + SxXY)*SzzYY + (-SyzYY - SxzXY)*SzYY)) + pow(SzXY,2)*(-SyXY*SzzYY - SxXX*SzzYY - SyzXY*SzYY - SxzXX*SzYY) + (SyzYY + SxzXY)*pow(SzXY,3))/denom;
      double c = (-SzzXY*(SyzXY*(SXX*SYY - pow(SXY,2)) + SxzXX*(SXX*SYY - pow(SXY,2)) + SzXX*(-SyXY*SYY - SxXX*SYY + (SyYY + SxXY)*SXY) + SzXY*((SyXY + SxXX)*SXY + (-SyYY - SxXY)*SXX)) - SzzXX*(SXX*(-SyzYY*SYY - SxzXY*SYY + (SyYY + SxXY)*SzYY) + SzXY*(SyXY*SYY + SxXX*SYY + (-SyYY - SxXY)*SXY) + (SyzYY + SxzXY)*pow(SXY,2) + (-SyXY*SzYY - SxXX*SzYY)*SXY) - SzXY*(SzXX*(-SyzXY*SYY - SxzXX*SYY + (-2*SyzYY - 2*SxzXY)*SXY + SyXY*SzYY + SxXX*SzYY) - SyzXY*SzYY*SXX - SxzXX*SzYY*SXX) - pow(SzXX,2)*(SyzYY*SYY + SxzXY*SYY + (-SyYY - SxXY)*SzYY) - SzXX*(SyzXY*SzYY*SXY + SxzXX*SzYY*SXY) - pow(SzXY,2)*(SyzXY*SXY + SxzXX*SXY + (SyzYY + SxzXY)*SXX + (SyYY + SxXY)*SzXX) - (-SyXY - SxXX)*pow(SzXY,3))/denom;
      double d = (SzzXX*(SzXY*((SyzYY + SxzXY)*SXY + SyXY*SzYY + SxXX*SzYY) + (-SyXY*SzzYY - SxXX*SzzYY)*SXY + ((SyYY + SxXY)*SzzYY + (-SyzYY - SxzXY)*SzYY)*SXX + (-SyYY - SxXY)*pow(SzXY,2)) + SzzXY*(SzXX*((-SyzYY - SxzXY)*SXY - SyXY*SzYY - SxXX*SzYY) + SzXY*(-SyzXY*SXY - SxzXX*SXY + (SyzYY + SxzXY)*SXX + (2*SyYY + 2*SxXY)*SzXX) + SyzXY*SzYY*SXX + SxzXX*SzYY*SXX + (-SyXY - SxXX)*pow(SzXY,2)) + SzXX*(SyzXY*SzzYY*SXY + SxzXX*SzzYY*SXY) + pow(SzzXY,2)*((SyXY + SxXX)*SXY + (-SyYY - SxXY)*SXX) + SzXY*(-SyzXY*SzzYY*SXX - SxzXX*SzzYY*SXX + SzXX*(SyXY*SzzYY + SxXX*SzzYY - SyzXY*SzYY - SxzXX*SzYY)) + pow(SzXX,2)*((-SyYY - SxXY)*SzzYY + (SyzYY + SxzXY)*SzYY) + (SyzXY + SxzXX)*pow(SzXY,3) + (-SyzYY - SxzXY)*SzXX*pow(SzXY,2))/denom;

      GlobalVector momentum = GlobalVector(a, c, 1.) / sqrt(pow(a,2) + pow(c,2) + 1.) * 1000;
      GlobalVector fitslope = GlobalVector(a, c, 1.);
      GlobalPoint fitpoint(b, d, 0.);

      // un-point from the direction of fitting
      momentum = GlobalVector(coordrot.xx() * momentum.x() + coordrot.yx() * momentum.y() + coordrot.zx() * momentum.z(),
			      coordrot.xy() * momentum.x() + coordrot.yy() * momentum.y() + coordrot.zy() * momentum.z(),
			      coordrot.xz() * momentum.x() + coordrot.yz() * momentum.y() + coordrot.zz() * momentum.z());

      fitslope = GlobalVector(coordrot.xx() * fitslope.x() + coordrot.yx() * fitslope.y() + coordrot.zx() * fitslope.z(),
			      coordrot.xy() * fitslope.x() + coordrot.yy() * fitslope.y() + coordrot.zy() * fitslope.z(),
			      coordrot.xz() * fitslope.x() + coordrot.yz() * fitslope.y() + coordrot.zz() * fitslope.z());
      fitslope /= fitslope.z();

      fitpoint = GlobalPoint(coordrot.xx() * fitpoint.x() + coordrot.yx() * fitpoint.y() + coordrot.zx() * fitpoint.z(),
			     coordrot.xy() * fitpoint.x() + coordrot.yy() * fitpoint.y() + coordrot.zy() * fitpoint.z(),
			     coordrot.xz() * fitpoint.x() + coordrot.yz() * fitpoint.y() + coordrot.zz() * fitpoint.z());

      double chi2 = 0.;
      int dof = 0;
      std::vector<TrajectoryMeasurement::ConstRecHitPointer>::const_iterator transHitPtr = transHits.begin();
      std::vector<const TrackingRecHit*>::const_iterator hit = hits.begin();
      std::vector<double>::const_iterator xi = listx.begin();
      std::vector<double>::const_iterator yi = listy.begin();
      std::vector<double>::const_iterator zi = listz.begin();
      std::vector<double>::const_iterator XX = listXX.begin();
      std::vector<double>::const_iterator XY = listXY.begin();
      std::vector<double>::const_iterator YY = listYY.begin();

      for (;  hit != hits.end();  ++transHitPtr, ++hit, ++xi, ++yi, ++zi, ++XX, ++XY, ++YY) {
	 DetId id = (*hit)->geographicalId();

	 GlobalPoint realspace;
	 GlobalPoint realhit;
	 // DTs: seek to the z (surface) and untransform
	 if (id.det() == DetId::Muon  &&  id.subdetId() == MuonSubdetId::DT) {
	    realspace = GlobalPoint(b, d, 0.) + GlobalVector(a, c, 1.) * (*zi);
	    realspace = GlobalPoint(coordrot.xx() * realspace.x() + coordrot.yx() * realspace.y() + coordrot.zx() * realspace.z(),
				    coordrot.xy() * realspace.x() + coordrot.yy() * realspace.y() + coordrot.zy() * realspace.z(),
				    coordrot.xz() * realspace.x() + coordrot.yz() * realspace.y() + coordrot.zz() * realspace.z());
	    realhit = dtGeometry->idToDet(id)->toGlobal((*hit)->localPosition());
	 }
	 // CSCs: untransform and seek to the real z position (surface)
	 else if (id.det() == DetId::Muon  &&  id.subdetId() == MuonSubdetId::CSC) {
	    double realzi = cscGeometry->idToDet(id)->toGlobal(LocalPoint((*hit)->localPosition())).z();
	    realspace = fitpoint + fitslope * (realzi - fitpoint.z());
	    realhit = cscGeometry->idToDet(id)->toGlobal((*hit)->localPosition());

//  	    CSCDetId cscid(id.rawId());
//  	    if (cscid.station() == 1) std::cout << "REALSPACE " << cscid << " " << realspace << " - " << realhit << " = " << (realspace - realhit) << std::endl;
	 }
	 else assert(false);
	 
	 if (realspace.mag() < 1e10) {} else return;  // quick fix for NaNs

	 GlobalTrajectoryParameters globalTrajectoryParameters(realspace, momentum, 1, &*magneticField);
	 AlgebraicSymMatrix66 error;
	 error(0,0) = error(1,1) = error(2,2) = error(3,3) = error(4,4) = error(5,5) = 1e-6;
	 
	 clonedHits.push_back((*hit)->clone());
	 TSOSes.push_back(TrajectoryStateOnSurface(globalTrajectoryParameters, CartesianTrajectoryError(error),
						   id.subdetId() == MuonSubdetId::DT ? dtGeometry->idToDet(id)->surface() : cscGeometry->idToDet(id)->surface()));

	 // calculate the chi2
	 bool include_in_fit = true;
	 if (m_isDT) include_in_fit = (id.det() == DetId::Muon  &&  id.subdetId() == MuonSubdetId::DT);
	 else include_in_fit = (id.det() == DetId::Muon  &&  id.subdetId() == MuonSubdetId::CSC);

	 double x = a * (*zi) + b;
	 double y = c * (*zi) + d;
	 double chi2i = (x - (*xi))*(x - (*xi))*(*XX) + 2*(x - (*xi))*(y - (*yi))*(*XY) + (y - (*yi))*(y - (*yi))*(*YY);
	 if (include_in_fit) {
	    chi2 += chi2i;
	    if ((*hit)->geographicalId().subdetId() == MuonSubdetId::DT)
	       dof += 1;
	    else
	       dof += 2; 
	 }
      } // end loop over hits
      dof -= 4;

      if (dof > 0) {
	 m_redchi2_10->Fill(chi2 / dof);
	 m_redchi2_100->Fill(chi2 / dof);
	 m_redchi2_1000->Fill(chi2 / dof);
      }

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

	 // Remember which Trajectory is associated with which Track
	 trajCounter++;
	 reference_map[trajCounter] = trackCounter;

      } // end if there are any clonedHits/TSOSes to work with

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
}

// ------------ method called once each job just before starting event loop  ------------
void 
ZeroFieldStandAloneMuonHIPRefitter::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
ZeroFieldStandAloneMuonHIPRefitter::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(ZeroFieldStandAloneMuonHIPRefitter);
