/** \class MuonTrackFinder
 *  Concrete Track finder for the Muon Reco
 *
 *  $Date: 2006/06/12 13:44:29 $
 *  $Revision: 1.6 $
 *  \author R. Bellan - INFN Torino
 */

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "TrackingTools/PatternTools/interface/TSCPBuilderNoMaterial.h"

#include "RecoMuon/TrackingTools/interface/MuonTrackFinder.h"
#include "RecoMuon/TrackingTools/interface/MuonTrajectoryBuilder.h"
#include "RecoMuon/TrackingTools/interface/MuonTrajectoryCleaner.h"

using namespace std;

// constructor
MuonTrackFinder::MuonTrackFinder(MuonTrajectoryBuilder *ConcreteMuonTrajectoryBuilder):
  theTrajBuilder(ConcreteMuonTrajectoryBuilder)
{
  theTrajCleaner = new MuonTrajectoryCleaner();
}

// Destructor
MuonTrackFinder::~MuonTrackFinder(){
  delete theTrajBuilder;
  delete theTrajCleaner;
};

// percolate the event setup
void MuonTrackFinder::setES(const edm::EventSetup &eSetup){
  theTrajBuilder->setES(eSetup);
}

// percolate the event setup
void MuonTrackFinder::setEvent(const edm::Event &event){
  theTrajBuilder->setEvent(event);
}


// Reconstruct trajectories
auto_ptr<reco::TrackCollection> MuonTrackFinder::reconstruct(const edm::Handle<TrajectorySeedCollection>& seeds){

  // Traj container
  TrajectoryContainer muonTrajectories;
  
  // reconstruct the traj
  for(TrajectorySeedCollection::const_iterator seed = seeds->begin();
      seed != seeds->end(); seed++){
    TrajectoryContainer muonTrajs_temp = theTrajBuilder->trajectories(*seed);
    for(TrajectoryContainer::const_iterator it = muonTrajs_temp.begin(); 
	it != muonTrajs_temp.end(); it++) 
      muonTrajectories.push_back(*it); 
  }

  // clean the clone traj
  theTrajCleaner->clean(muonTrajectories); //used by reference...
  return convert(muonTrajectories);
}
  

// Convert trajectory container in ...
auto_ptr<reco::TrackCollection> MuonTrackFinder::convert(TrajectoryContainer &cont) const{
  
  // FIXME??
  auto_ptr<reco::TrackCollection> muons;

  edm::Handle<reco::TrackExtraCollection> trackExtraCollection;
  //reco::TrackExtraCollection trackExtraCollection;
 
  int counter = 0;

  for(TrajectoryContainer::const_iterator it = cont.begin();
      it != cont.end(); ++it){
    // Get the last TrajectoryMeasurement

    // convert/build a track starting from a trajectory 
    reco::Track track = buildTrack(*it);
    
    // build the track extra in the track
    //    trackExtraCollection->push_back( buildTrackExtra( track, (*it) ) );

    // build the track extra ref in the track
    // reco::TrackExtraRef trackExtraRef(trackExtraCollection,counter);
    
    // set the track extra ref in the track
    // track.setExtra(trackExtraRef);

    muons->push_back(track);

  }
  return muons;
}

// algoResults == trajectoy
reco::Track MuonTrackFinder::buildTrack (const Trajectory& trajectory) const {
  
  // FIXME: check the prop direction
  TrajectoryStateOnSurface innerTSOS;

  if (trajectory.direction() == alongMomentum) {
    innerTSOS = trajectory.firstMeasurement().updatedState();
  } else { 
    innerTSOS = trajectory.lastMeasurement().updatedState();
  }
  
    
  // This is needed to extrapolate the tsos at vertex
  // FIXME: check it!
  TSCPBuilderNoMaterial tscpBuilder;
  TrajectoryStateClosestToPoint tscp = tscpBuilder( innerTSOS,
						    GlobalPoint(0,0,0) );//FIXME Correct?
  
  reco::perigee::Parameters param = tscp.perigeeParameters();
  reco::perigee::Covariance covar = tscp.perigeeError();


  const Trajectory::RecHitContainer& transRecHits = trajectory.recHits();
  
  int dof=0;

  for(Trajectory::RecHitContainer::const_iterator rechit = transRecHits.begin();
      rechit != transRecHits.end(); ++rechit)
    if ((*rechit).isValid()) dof += (*rechit).dimension();
  
  unsigned short int ndof = max(dof - 5, 0);
  
  reco::Track track(trajectory.chiSquared(), 
		    trajectory.foundHits(),//FIXME to be fixed in Trajectory.h
		    0, //FIXME no corresponding method in trajectory.h
		    trajectory.lostHits(),//FIXME to be fixed in Trajectory.h
		    ndof, param, covar);
  return track;
}


reco::TrackExtra MuonTrackFinder::buildTrackExtra(reco::Track &track, const Trajectory& trajectory) const{

  const Trajectory::RecHitContainer transRecHits = trajectory.recHits();
  
  // put the collection of TrackingRecHit in the event
  
  // sets the outermost and innermost TSOSs
  // FIXME: check it!
  TrajectoryStateOnSurface outerTSOS;
  TrajectoryStateOnSurface innerTSOS;
  
  if (trajectory.direction() == alongMomentum) {
    outerTSOS = trajectory.lastMeasurement().updatedState();
    innerTSOS = trajectory.firstMeasurement().updatedState();
  } else { 
      outerTSOS = trajectory.firstMeasurement().updatedState();
      innerTSOS = trajectory.lastMeasurement().updatedState();
  }
  
  //build the TrackExtra
  GlobalPoint v = outerTSOS.globalParameters().position();
  GlobalVector p = outerTSOS.globalParameters().momentum();
  math::XYZVector outmom( p.x(), p.y(), p.z() );
  math::XYZPoint  outpos( v.x(), v.y(), v.z() );   

  reco::TrackExtra trackExtra(outpos, outmom, true);
  
  return trackExtra;
  
//   for(Trajectory::RecHitContainer::const_iterator recHit = transRecHits.begin();
//       recHit != transHits.end(); ++recHit)
//     trackExtra.add(TrackingRecHitRef( *recHit ) );
  
//   //create a TrackExtraRef
//   reco::TrackExtraRef  trackExtraRef( trackExtra);
  
//   //use the TrackExtraRef to assign the TrackExtra to the Track
//   track.setExtra(trackExtraRef);
  
}


