// -*- C++ -*-
//
// Package:    CSCOverlapsTrackPreparation
// Class:      CSCOverlapsTrackPreparation
// 
/**\class CSCOverlapsTrackPreparation CSCOverlapsTrackPreparation.cc Alignment/CSCOverlapsTrackPreparation/src/CSCOverlapsTrackPreparation.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jim Pivarski
//         Created:  Wed Dec 12 13:31:55 CST 2007
// $Id: CSCOverlapsTrackPreparation.cc,v 1.5 2010/03/25 00:59:12 pivarski Exp $
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
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TTree.h"

// products
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

//
// class decleration
//

class CSCOverlapsTrackPreparation : public edm::EDProducer {
   public:
      explicit CSCOverlapsTrackPreparation(const edm::ParameterSet&);
      ~CSCOverlapsTrackPreparation();

   private:
      enum {kNothing, kSimpleFit, kAllButOne, kExtrapolate};

      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      // ----------member data ---------------------------
      edm::InputTag m_src;
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
CSCOverlapsTrackPreparation::CSCOverlapsTrackPreparation(const edm::ParameterSet& iConfig)
   : m_src(iConfig.getParameter<edm::InputTag>("src"))
{
   produces<std::vector<Trajectory> >();
   produces<TrajTrackAssociationCollection>();
}


CSCOverlapsTrackPreparation::~CSCOverlapsTrackPreparation()
{
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
CSCOverlapsTrackPreparation::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // fiduciality cuts
  const double cut_ME11 = 0.086;
  const double cut_ME12 = 0.090;
  const double cut_MEx1 = 0.1790;
  const double cut_MEx2 = 0.0905;

  edm::Handle<reco::TrackCollection> tracks;
  iEvent.getByLabel(m_src, tracks);

  edm::ESHandle<CSCGeometry> cscGeometry;
  iSetup.get<MuonGeometryRecord>().get(cscGeometry);

  edm::ESHandle<MagneticField> magneticField;
  iSetup.get<IdealMagneticFieldRecord>().get(magneticField);

  edm::ESHandle<GlobalTrackingGeometry> globalGeometry;
  iSetup.get<GlobalTrackingGeometryRecord>().get(globalGeometry);

  TrajectoryStateTransform transformer;
  MuonTransientTrackingRecHitBuilder muonTransBuilder;

  // Create a collection of Trajectories, to put in the Event
  std::auto_ptr<std::vector<Trajectory> > trajectoryCollection(new std::vector<Trajectory>);

  // Remember which trajectory is associated with which track
  std::map<edm::Ref<std::vector<Trajectory> >::key_type, edm::Ref<reco::TrackCollection>::key_type> reference_map;
  edm::Ref<std::vector<Trajectory> >::key_type trajCounter = 0;
  edm::Ref<reco::TrackCollection>::key_type trackCounter = 0;

  for (reco::TrackCollection::const_iterator track = tracks->begin();  track != tracks->end();  ++track) {
    trackCounter++;

    // linear fits for a fiduciality cut: "x" is local z and "y" is localphi
    std::vector<CSCDetId> keys;
    std::map<CSCDetId,double> sum_w;
    std::map<CSCDetId,double> sum_x;
    std::map<CSCDetId,double> sum_xx;
    std::map<CSCDetId,double> sum_y;
    std::map<CSCDetId,double> sum_xy;
    for (trackingRecHit_iterator hit = track->recHitsBegin();  hit != track->recHitsEnd();  ++hit) {
      DetId id = (*hit)->geographicalId();
      if (id.det() == DetId::Muon  &&  id.subdetId() == MuonSubdetId::CSC) {
	CSCDetId cscid(id.rawId());
	CSCDetId chamberid(cscid.endcap(), cscid.station(), cscid.ring(), cscid.chamber(), 0);

	int strip = cscGeometry->layer(id)->geometry()->nearestStrip((*hit)->localPosition());
	double angle = cscGeometry->layer(id)->geometry()->stripAngle(strip) - M_PI/2.;
	double sinAngle = sin(angle);
	double cosAngle = cos(angle);

	double localphi = atan2((*hit)->localPosition().x(), cscGeometry->idToDet(id)->toGlobal((*hit)->localPosition()).perp());
	double z = cscGeometry->idToDet(chamberid)->toLocal(cscGeometry->idToDet(id)->toGlobal((*hit)->localPosition())).z();

	double xx = (*hit)->localPositionError().xx();
	double xy = (*hit)->localPositionError().xy();
	double yy = (*hit)->localPositionError().yy();
	double err2 = xx*cosAngle*cosAngle + 2.*xy*sinAngle*cosAngle + yy*sinAngle*sinAngle;
	  
	if (sum_w.find(chamberid) == sum_w.end()) {
	  keys.push_back(chamberid);
	  sum_w[chamberid] = 0.;
	  sum_x[chamberid] = 0.;
	  sum_xx[chamberid] = 0.;
	  sum_y[chamberid] = 0.;
	  sum_xy[chamberid] = 0.;
	}
	sum_w[chamberid] += 1./err2;
	sum_x[chamberid] += z/err2;
	sum_xx[chamberid] += z*z/err2;
	sum_y[chamberid] += localphi/err2;
	sum_xy[chamberid] += z*localphi/err2;
	  
      } // end if CSC
    } // end first loop over hits

      // do the fits and determine whether the endpoints are inside the fiducial region
    std::map<CSCDetId,bool> okay;
    for (std::vector<CSCDetId>::const_iterator key = keys.begin();  key != keys.end();  ++key) {
      double delta = (sum_w[*key] * sum_xx[*key]) - (sum_x[*key] * sum_x[*key]);
      double intercept = ((sum_xx[*key] * sum_y[*key]) - (sum_x[*key] * sum_xy[*key]))/delta;
      double slope = ((sum_w[*key] * sum_xy[*key]) - (sum_x[*key] * sum_y[*key]))/delta;

      CSCDetId layer1 = CSCDetId(key->endcap(), key->station(), key->ring(), key->chamber(), 1);
      CSCDetId layer6 = CSCDetId(key->endcap(), key->station(), key->ring(), key->chamber(), 6);
      double z1 = cscGeometry->idToDet(*key)->toLocal(cscGeometry->idToDet(layer1)->toGlobal(LocalPoint(0., 0., 0.))).z();
      double z6 = cscGeometry->idToDet(*key)->toLocal(cscGeometry->idToDet(layer6)->toGlobal(LocalPoint(0., 0., 0.))).z();

      double localphi1 = intercept + slope*z1;
      double localphi6 = intercept + slope*z6;

      if (key->station() == 1  &&  (key->ring() == 1  ||  key->ring() == 4)) {
	okay[*key] = (fabs(localphi1) < cut_ME11  &&  fabs(localphi6) < cut_ME11);
      }
      else if (key->station() == 1  &&  key->ring() == 2) {
	okay[*key] = (fabs(localphi1) < cut_ME12  &&  fabs(localphi6) < cut_ME12);
      }
      else if (key->station() > 1  &&  key->ring() == 1) {
	okay[*key] = (fabs(localphi1) < cut_MEx1  &&  fabs(localphi6) < cut_MEx1);
      }
      else if (key->station() > 1  &&  key->ring() == 2) {
	okay[*key] = (fabs(localphi1) < cut_MEx2  &&  fabs(localphi6) < cut_MEx2);
      }
      else {
	okay[*key] = false;
      }

    } // end loop over hit chambers

      // now we'll actually put hits on the new trajectory
      // these must be in lock-step
    edm::OwnVector<TrackingRecHit> clonedHits;
    std::vector<TrajectoryMeasurement::ConstRecHitPointer> transHits;
    std::vector<TrajectoryStateOnSurface> TSOSes;

    for (trackingRecHit_iterator hit = track->recHitsBegin();  hit != track->recHitsEnd();  ++hit) {
      DetId id = (*hit)->geographicalId();
      if (id.det() == DetId::Muon  &&  id.subdetId() == MuonSubdetId::CSC) {
	CSCDetId cscid(id.rawId());
	CSCDetId chamberid(cscid.endcap(), cscid.station(), cscid.ring(), cscid.chamber(), 0);

	if (okay.find(chamberid) != okay.end()  &&  okay[chamberid]) {
	  double localphi = atan2((*hit)->localPosition().x(), cscGeometry->idToDet(id)->toGlobal((*hit)->localPosition()).perp());
	  if ((chamberid.station() == 1  &&  (chamberid.ring() == 1  ||  chamberid.ring() == 4)  &&  fabs(localphi) < cut_ME11)  ||
	      (chamberid.station() == 1  &&  chamberid.ring() == 2  &&  fabs(localphi) < cut_ME12)  ||
	      (chamberid.station() > 1  &&  chamberid.ring() == 1  &&  fabs(localphi) < cut_MEx1)  ||
	      (chamberid.station() > 1  &&  chamberid.ring() == 2  &&  fabs(localphi) < cut_MEx2)) {

	    const Surface &layerSurface = cscGeometry->idToDet(id)->surface();
	    TrajectoryMeasurement::ConstRecHitPointer hitPtr(muonTransBuilder.build(&**hit, globalGeometry));

	    AlgebraicVector params(5);   // meaningless, CSCOverlapsAlignmentAlgorithm does the fit internally
	    params[0] = 1.;  // straight-forward direction
	    params[1] = 0.;
	    params[2] = 0.;
	    params[3] = 0.;  // center of the chamber
	    params[4] = 0.;
	    LocalTrajectoryParameters localTrajectoryParameters(params, 1., false);
	    LocalTrajectoryError localTrajectoryError(0.001, 0.001, 0.001, 0.001, 0.001);

	    // these must be in lock-step
	    clonedHits.push_back((*hit)->clone());
	    transHits.push_back(hitPtr);
	    TSOSes.push_back(TrajectoryStateOnSurface(localTrajectoryParameters, localTrajectoryError, layerSurface, &*magneticField));

	  } // end hit fiduciality cut
	} // end track fiduciality cut
      } // end if CSC
    } // end loop over hits

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
void CSCOverlapsTrackPreparation::beginJob() {}

// ------------ method called once each job just after ending the event loop  ------------
void CSCOverlapsTrackPreparation::endJob() {}

//define this as a plug-in
DEFINE_FWK_MODULE(CSCOverlapsTrackPreparation);
