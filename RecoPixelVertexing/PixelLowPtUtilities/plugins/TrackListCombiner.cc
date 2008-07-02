#include "TrackListCombiner.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"

/*****************************************************************************/
TrackListCombiner::TrackListCombiner(const edm::ParameterSet& ps)
{
  trackProducers = ps.getParameter<vector<string> >("trackProducers");

  produces<reco::TrackCollection>();
  produces<vector<Trajectory> >();
}

/*****************************************************************************/
TrackListCombiner::~TrackListCombiner()
{
}

/*****************************************************************************/
void TrackListCombiner::produce(edm::Event& ev, const edm::EventSetup& es)
{
  auto_ptr<reco::TrackCollection> outputTracks(new reco::TrackCollection);
  auto_ptr<vector<Trajectory> >   outputTrajes(new vector<Trajectory>   );

  LogTrace("MinBiasTracking")
    << "[TrackListCombiner]";

  // Go through all track producers
  int i = 1;
  for(vector<string>::iterator trackProducer = trackProducers.begin();
                               trackProducer!= trackProducers.end();
                               trackProducer++, i++)
  {
    reco::TrackBase::TrackAlgorithm algo;
    switch(i) 
    {
      case 1:  algo = reco::TrackBase::iter1; break;
      case 2:  algo = reco::TrackBase::iter2; break;
      case 3:  algo = reco::TrackBase::iter3; break;
      default: algo = reco::TrackBase::undefAlgorithm;
    }

    // Get track collection
    edm::Handle<reco::TrackCollection> trackHandle;
    ev.getByLabel(*trackProducer,      trackHandle);
    const reco::TrackCollection & trackCollection = *(trackHandle.product());

    // Get trajectory collection
    edm::Handle<vector<Trajectory> > trajeHandle;
    ev.getByLabel(*trackProducer,    trajeHandle);
    const vector<Trajectory> & trajeCollection = *(trajeHandle.product());

    LogTrace("MinBiasTracking")
      << " [TrackListCombiner] " << *trackProducer
      << " : " << trackCollection.size() << "|" << trajeCollection.size();

    for(reco::TrackCollection::const_iterator track = trackCollection.begin();
                                              track!= trackCollection.end();
                                              track++)
    {
      // Get new track
      reco::Track * theTrack = new reco::Track(*track);

      // Set extra
      reco::TrackExtraRef theTrackExtraRef = track->extra();    
      theTrack->setExtra(theTrackExtraRef);    
      theTrack->setHitPattern((*theTrackExtraRef).recHits());
     
      // Set algorithm
      theTrack->setAlgorithm(algo);

      // Store track
      outputTracks->push_back(*theTrack);

      delete theTrack;
    }

    for(vector<Trajectory>::const_iterator traje = trajeCollection.begin();
                                           traje!= trajeCollection.end();
                                           traje++)
    {
      // Get new trajectory
      Trajectory * theTraje = new Trajectory(*traje);

      // Store trajectory
      outputTrajes->push_back(*theTraje);

      delete theTraje;
    }
  }

  LogTrace("MinBiasTracking")
    << " [TrackListCombiner] allTracks : " << outputTracks->size()
                                    << "|" << outputTrajes->size();

  // Put back result to event
  ev.put(outputTracks);
  ev.put(outputTrajes);
}
