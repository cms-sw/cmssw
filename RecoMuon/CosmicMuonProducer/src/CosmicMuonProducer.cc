#include "RecoMuon/CosmicMuonProducer/src/CosmicMuonProducer.h"
// Package:    CosmicMuonProducer
// Class:      CosmicMuonProducer
// 
/**\class CosmicMuonProducer
 *
 * Description: <one line class summary>
 *
 * Implementation:
 *
 * $Date:$
 * $Revision:$
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

#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
//
// constructors and destructor
//
CosmicMuonProducer::CosmicMuonProducer(const edm::ParameterSet& iConfig)
{
  produces<reco::TrackCollection>();
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
  std::vector<Trajectory> theTrajs = theBuilder.trajectories(iEvent,iSetup);

  edm::LogInfo("CosmicMuonProducer") << "No. of Trajectories: "<<theTrajs.size();

  std::auto_ptr<reco::TrackCollection> outputTColl(new reco::TrackCollection);

  for (std::vector<Trajectory>::const_iterator theT = theTrajs.begin(); theT != theTrajs.end(); theT++) {

    //sets the innermost TSOSs
    TrajectoryStateOnSurface innertsos;

    if (theT->direction() == alongMomentum) {
      TrajectoryMeasurement firstM = theT->firstMeasurement();

      innertsos = theT->firstMeasurement().updatedState();

    } else {
      innertsos = theT->lastMeasurement().updatedState();

    }
    reco::Track* theTrack;

    TSCPBuilderNoMaterial tscpBuilder;
    TrajectoryStateClosestToPoint tscp = tscpBuilder(innertsos,
                                                   GlobalPoint(0,0,0) );

    reco::perigee::Parameters param = tscp.perigeeParameters();
    reco::perigee::Covariance covar = tscp.perigeeError();

    int ndof = 0;
    const edm::OwnVector<const TransientTrackingRecHit>& rhL = theT->recHits();

    for (edm::OwnVector<const TransientTrackingRecHit>::const_iterator irh = rhL.begin(); irh != rhL.end(); irh++) {
       ndof += irh->dimension();
    }
    ndof -= 5;
    if (ndof < 0) ndof =0;
    edm::LogInfo("CosmicMuonProducer") << "ndof "<<ndof;
    edm::LogInfo("CosmicMuonProducer") << "chi2 "<<theT->chiSquared();

    theTrack = new reco::Track(theT->chiSquared(),
                               ndof,
                               theT->foundHits(),// number of rechit
                               0,
                               theT->lostHits(),
                               param,
                               covar);

    outputTColl->push_back(*theTrack);
  }

  iEvent.put(outputTColl);

}

//define this as a plug-in
DEFINE_FWK_MODULE(CosmicMuonProducer)
