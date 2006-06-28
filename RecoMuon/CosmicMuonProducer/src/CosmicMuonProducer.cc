#include "RecoMuon/CosmicMuonProducer/src/CosmicMuonProducer.h"

/**\class CosmicMuonProducer
 *
 * Description: CosmicMuonProducer
 *
 * Implementation:
 *
 * $Date: 2006/06/26 23:01:44 $
 * $Revision: 1.4 $
 * Original Author:  Chang Liu
 *        Created:  Tue Jun 13 02:46:17 CEST 2006
**/

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/PatternTools/interface/TrajectoryFitter.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "RecoMuon/CosmicMuonProducer/interface/CosmicMuonTrajectoryBuilder.h"
#include "TrackingTools/PatternTools/interface/TransverseImpactPointExtrapolator.h"
#include "TrackingTools/PatternTools/interface/TSCPBuilderNoMaterial.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateClosestToPoint.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "TrackingTools/TrajectoryParametrization/interface/PerigeeTrajectoryError.h"
#include "TrackingTools/TrajectoryParametrization/interface/PerigeeTrajectoryParameters.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"

#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
//
// constructors and destructor
//
CosmicMuonProducer::CosmicMuonProducer(const edm::ParameterSet& iConfig)
{
  produces<reco::TrackCollection>();
  produces<TrackingRecHitCollection>();
  produces<reco::TrackExtraCollection>();

  edm::LogInfo("CosmicMuonProducer")<<"cosmic begin";

}


CosmicMuonProducer::~CosmicMuonProducer()
{
 
  edm::LogInfo("CosmicMuonProducer")<<"cosmic end";

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
CosmicMuonProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  edm::LogInfo("CosmicMuonProducer") << "Analyzing event number: " << iEvent.id();
  edm::ESHandle<MagneticField> theMF;
  iSetup.get<IdealMagneticFieldRecord>().get(theMF);
  const MagneticField * field(&(*theMF));
  CosmicMuonTrajectoryBuilder theBuilder(field);
  const std::vector<Trajectory>& theTrajs = theBuilder.trajectories(iEvent,iSetup);

  edm::LogInfo("CosmicMuonProducer") << "No. of Trajectories: "<<theTrajs.size();

  std::auto_ptr<reco::TrackCollection> outputTColl(new reco::TrackCollection);
  std::auto_ptr<TrackingRecHitCollection> outputRHColl (new TrackingRecHitCollection);
  std::auto_ptr<reco::TrackExtraCollection> outputTEColl(new reco::TrackExtraCollection);

  for (std::vector<Trajectory>::const_iterator theT = theTrajs.begin(); theT != theTrajs.end(); theT++) {

    //sets the innermost TSOSs
    TrajectoryStateOnSurface innertsos;
    GlobalPoint v;
    GlobalVector p;

    if (theT->direction() == alongMomentum) {
      innertsos = theT->firstMeasurement().updatedState();
      v = theT->lastMeasurement().updatedState().globalPosition();
      p =theT->lastMeasurement().updatedState().globalMomentum();
    } else {
      innertsos = theT->lastMeasurement().updatedState();
      v = theT->firstMeasurement().updatedState().globalPosition();
      p = theT->firstMeasurement().updatedState().globalMomentum();
    }
    if (!innertsos.isValid()) continue;
    TSCPBuilderNoMaterial tscpBuilder;
    TrajectoryStateClosestToPoint tscp = tscpBuilder(innertsos,
                                                   GlobalPoint(0,0,0) );

    reco::perigee::Parameters param = tscp.perigeeParameters();
    reco::perigee::Covariance covar = tscp.perigeeError();

    int ndof = 0;
    const edm::OwnVector<const TransientTrackingRecHit>& rhL = theT->recHits();

    for (edm::OwnVector<const TransientTrackingRecHit>::const_iterator irh = rhL.begin(); irh != rhL.end(); irh++) {
       outputRHColl->push_back(((irh->hit())->clone()));
       ndof += irh->dimension();
    }
    edm::OrphanHandle<TrackingRecHitCollection> ohRH =iEvent.put(outputRHColl);

    ndof -= 5;
    if (ndof < 0) ndof =0;
    edm::LogInfo("CosmicMuonProducer") << "ndof "<<ndof;
    edm::LogInfo("CosmicMuonProducer") << "chi2 "<<theT->chiSquared();

    reco::Track  theTrack(theT->chiSquared(),
                               ndof,
                               theT->foundHits(),// number of rechit
                               0,
                               theT->lostHits(),
                               param,
                               covar);

     math::XYZVector outmom( p.x(), p.y(), p.z() );
     math::XYZPoint  outpos( v.x(), v.y(), v.z() );   
     reco::TrackExtra *theTrackExtra = new reco::TrackExtra(outpos, outmom, true);
     for(edm::OwnVector<const TransientTrackingRecHit>::const_iterator j=rhL.begin();
	    j!=rhL.end(); j++){
	  theTrackExtra->add(TrackingRecHitRef(ohRH,0));
	}

     outputTEColl->push_back(*theTrackExtra);
     edm::OrphanHandle<reco::TrackExtraCollection>ohTE=iEvent.put(outputTEColl);

     reco::TrackExtraRef  theTrackExtraRef(ohTE,0);
     theTrack.setExtra(theTrackExtraRef);
     outputTColl->push_back(theTrack);      

     iEvent.put(outputTColl);

  }

}

//define this as a plug-in
DEFINE_FWK_MODULE(CosmicMuonProducer)
