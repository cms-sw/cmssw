#include "RecoJets/JetAssociationProducers/interface/TrackExtrapolator.h"
#include "TrackingTools/TrackAssociator/interface/DetIdAssociator.h"


#include <vector>


//
// constructors and destructor
//
TrackExtrapolator::TrackExtrapolator(const edm::ParameterSet& iConfig) :
  tracksSrc_(iConfig.getParameter<edm::InputTag> ("trackSrc"))
{
  trackQuality_ = 
    reco::TrackBase::qualityByName (iConfig.getParameter<std::string> ("trackQuality"));
  if (trackQuality_ == reco::TrackBase::undefQuality) { // we have a problem
    throw cms::Exception("InvalidInput") << "Unknown trackQuality value '" 
					 << iConfig.getParameter<std::string> ("trackQuality")
					 << "'. See possible values in 'reco::TrackBase::qualityByName'";
  }

  produces< std::vector<reco::TrackExtrapolation> > ();
}


TrackExtrapolator::~TrackExtrapolator()
{
}


//
// member functions
//

// ------------ method called on each new Event  ------------
void
TrackExtrapolator::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  // get stuff from Event Setup
  edm::ESHandle<MagneticField> field_h;
  iSetup.get<IdealMagneticFieldRecord>().get(field_h);
  edm::ESHandle<Propagator> propagator_h;
  iSetup.get<TrackingComponentsRecord>().get("SteppingHelixPropagatorAlong", propagator_h);
  edm::ESHandle<DetIdAssociator> ecalDetIdAssociator_h;
  iSetup.get<DetIdAssociatorRecord>().get("EcalDetIdAssociator", ecalDetIdAssociator_h);
  FiducialVolume const & ecalvolume = ecalDetIdAssociator_h->volume();

  // get stuff from Event
  edm::Handle <reco::TrackCollection> tracks_h;
  iEvent.getByLabel (tracksSrc_, tracks_h);

  std::auto_ptr< std::vector<reco::TrackExtrapolation> > extrapolations( new std::vector<reco::TrackExtrapolation>() );

  // Get list of tracks we want to extrapolate
  std::vector <reco::TrackRef> goodTracks;
  for ( reco::TrackCollection::const_iterator trkBegin = tracks_h->begin(),
	  trkEnd = tracks_h->end(), itrk = trkBegin;
	itrk != trkEnd; ++itrk ) {
    reco::TrackBase::TrackQuality trackQuality = reco::TrackBase::TrackQuality (trackQuality_);

    // Cut on track quality
    if (itrk->quality (trackQuality)) {
      goodTracks.push_back (reco::TrackRef (tracks_h, itrk - trkBegin));
    }
  }
  std::vector<reco::TrackBase::Point>  vresultPos(1);
  std::vector<reco::TrackBase::Vector> vresultMom(1);
  

  // Now loop through the list of tracks and extrapolate them
  for ( std::vector<reco::TrackRef>::const_iterator trkBegin = goodTracks.begin(),
	  trkEnd = goodTracks.end(), itrk = trkBegin; 
	itrk != trkEnd; ++itrk ) {
    if( propagateTrackToVolume( **itrk, *field_h, *propagator_h, ecalvolume,
				vresultPos[0], vresultMom[0]) ) {
      extrapolations->push_back( reco::TrackExtrapolation( *itrk, 
							   vresultPos, 
							   vresultMom ) );
    }
  }
  iEvent.put( extrapolations );
}

// ------------ method called once each job just before starting event loop  ------------
void 
TrackExtrapolator::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
TrackExtrapolator::endJob() {
}




// -----------------------------------------------------------------------------
//
bool TrackExtrapolator::propagateTrackToVolume( const reco::Track& fTrack,
						const MagneticField& fField,
						const Propagator& fPropagator,
						const FiducialVolume& volume,
						reco::TrackBase::Point & resultPos,
						reco::TrackBase::Vector & resultMom
						)
{
  GlobalPoint trackPosition (fTrack.vx(), fTrack.vy(), fTrack.vz()); // reference point
  GlobalVector trackMomentum (fTrack.px(), fTrack.py(), fTrack.pz()); // reference momentum
  if (fTrack.extra().isAvailable() ) { // use outer point information, if available
    trackPosition =  GlobalPoint (fTrack.outerX(), fTrack.outerY(), fTrack.outerZ());
    trackMomentum = GlobalVector (fTrack.outerPx(), fTrack.outerPy(), fTrack.outerPz());
  }

  GlobalTrajectoryParameters trackParams(trackPosition, trackMomentum, fTrack.charge(), &fField);
  FreeTrajectoryState trackState (trackParams);

  TrajectoryStateOnSurface 
    propagatedInfo = fPropagator.propagate (trackState, 
					    *Cylinder::build (volume.minR(), Surface::PositionType (0,0,0),
							      Surface::RotationType()
							     )
					    );

  // if the track went through either side of the endcaps, repropagate the track
  double minz=volume.minZ();
  if(propagatedInfo.isValid() && propagatedInfo.globalPosition().z()>minz) {
    propagatedInfo = fPropagator.propagate (trackState, 
					    *Plane::build (Surface::PositionType (0,0,minz),
							   Surface::RotationType())
					    );

  } else if(propagatedInfo.isValid() && propagatedInfo.globalPosition().z()<-minz) {
    propagatedInfo = fPropagator.propagate (trackState, 
					    *Plane::build (Surface::PositionType (0,0,-minz),
							   Surface::RotationType())
					    );
  }
  

  if (propagatedInfo.isValid()) {
    resultPos = propagatedInfo.globalPosition ();
    resultMom = propagatedInfo.globalMomentum ();
    return true;
  }
  else { 
    return false;
  }
}




//define this as a plug-in
DEFINE_FWK_MODULE(TrackExtrapolator);
