#include "RecoJets/JetAssociationProducers/interface/TrackExtrapolator.h"



#include <vector>


//
// constructors and destructor
//
TrackExtrapolator::TrackExtrapolator(const edm::ParameterSet& iConfig) :
  tracksSrc_(iConfig.getParameter<edm::InputTag> ("trackSrc")),
  radii_(iConfig.getParameter<std::vector<double> > ("radii") )
{
  trackQuality_ = 
    reco::TrackBase::qualityByName (iConfig.getParameter<std::string> ("trackQuality"));
  if (trackQuality_ == reco::TrackBase::undefQuality) { // we have a problem
    throw cms::Exception("InvalidInput") << "Unknown trackQuality value '" 
					 << iConfig.getParameter<std::string> ("trackQuality")
					 << "'. See possible values in 'reco::TrackBase::qualityByName'";
  }

  std::sort( radii_.begin(), radii_.end() );
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
  
  // Now loop through the list of tracks and extrapolate them. 
  // At each radius that's desired, store the extrapolation.
  // Then add the extrapolation to the list, and write to the event. 
  for ( std::vector<reco::TrackRef>::const_iterator trkBegin = goodTracks.begin(),
	  trkEnd = goodTracks.end(), itrk = trkBegin; 
	itrk != trkEnd; ++itrk ) {
    std::vector<reco::TrackBase::Point>  vresultPos;
    std::vector<reco::TrackBase::Vector> vresultMom;
    std::vector<reco::TrackBase::Vector> vresultDir;
    std::vector<bool> visValid;
    for ( std::vector<double>::const_iterator radBegin = radii_.begin(),
	    radEnd = radii_.end(), ir = radBegin;
	  ir != radEnd; ++ir ) {
      reco::TrackBase::Point resultPos;
      reco::TrackBase::Vector resultMom;
      reco::TrackBase::Vector resultDir;
      bool isValid = propagateTrackToR( **itrk, *field_h, *propagator_h, *ir, 
					resultPos, resultMom, resultDir );
      visValid.push_back(isValid);
      vresultPos.push_back( resultPos );
      vresultMom.push_back( resultMom );
      vresultDir.push_back( resultDir );
    }
    extrapolations->push_back( reco::TrackExtrapolation( *itrk, 
							 visValid, 
							 vresultPos, 
							 vresultMom, 
							 vresultDir ) );
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
bool TrackExtrapolator::propagateTrackToR( const reco::Track& fTrack,
					   const MagneticField& fField,
					   const Propagator& fPropagator,
					   const double & R,
					   reco::TrackBase::Point & resultPos,
					   reco::TrackBase::Vector & resultMom,
					   reco::TrackBase::Vector & resultDir
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
					    *Cylinder::build (Surface::PositionType (0,0,0),
							      Surface::RotationType(),
							      R)
					    );
  if (propagatedInfo.isValid()) {
    resultPos = propagatedInfo.globalPosition ();
    resultMom = propagatedInfo.globalMomentum ();
    resultDir = propagatedInfo.globalDirection(); 
    return true;
  }
  else { 
    return false;
  }
}




//define this as a plug-in
DEFINE_FWK_MODULE(TrackExtrapolator);
