#include "FastSimulation/Tracking/plugins/ConversionTrackRefFix.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EgammaTrackReco/interface/ConversionTrack.h"

using namespace std;
using namespace reco;
using namespace edm;

ConversionTrackRefFix::ConversionTrackRefFix(const edm::ParameterSet& iConfig)
{
  InputTag conversionTracksTag = iConfig.getParameter<InputTag>("src");
  InputTag newTracksTag = iConfig.getParameter<InputTag>("newTrackCollection");
  
  produces<ConversionTrackCollection>();

  conversionTracksToken = consumes<ConversionTrackCollection>(conversionTracksTag);
  newTracksToken = consumes<TrackCollection>(newTracksTag);
}

ConversionTrackRefFix::~ConversionTrackRefFix(){}


void
ConversionTrackRefFix::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  Handle<ConversionTrackCollection> conversionTracks;
  iEvent.getByToken(conversionTracksToken,conversionTracks);

  Handle<TrackCollection> newTracks;
  iEvent.getByToken(newTracksToken,newTracks);

  auto_ptr<ConversionTrackCollection> output(new ConversionTrackCollection);
  
  for(const ConversionTrack &  conversion : *(conversionTracks.product())){
    size_t trackIndex = conversion.trackRef().key();
    output->push_back(ConversionTrack(TrackBaseRef(TrackRef(newTracks,trackIndex))));
    output->back().setTrajRef(conversion.trajRef());
    output->back().setIsTrackerOnly(conversion.isTrackerOnly());
    output->back().setIsArbitratedEcalSeeded(conversion.isArbitratedEcalSeeded());
    output->back().setIsArbitratedMerged(conversion.isArbitratedMerged());
    output->back().setIsArbitratedMergedEcalGeneral(conversion.isArbitratedMergedEcalGeneral());
  }

  iEvent.put(output);

}
