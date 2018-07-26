/****************************************************************************
 *
 * This is a part of CTPPS offline software.
 * Authors:
 *   Laurent Forthomme (laurent.forthomme@cern.ch)
 *   Nicola Minafra (nicola.minafra@cern.ch)
 *   Mateusz Szpyrka (mateusz.szpyrka@cern.ch)
 *
 ****************************************************************************/

#include <memory>
#include <utility>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/Common/interface/DetSetVector.h"

#include "DataFormats/CTPPSDetId/interface/TotemTimingDetId.h"
#include "DataFormats/CTPPSReco/interface/TotemTimingRecHit.h"
#include "DataFormats/CTPPSReco/interface/TotemTimingLocalTrack.h"

#include "RecoCTPPS/TotemRPLocal/interface/TotemTimingTrackRecognition.h"

class TotemTimingLocalTrackFitter : public edm::stream::EDProducer<>
{
  public:
    explicit TotemTimingLocalTrackFitter( const edm::ParameterSet& );

    static void fillDescriptions( edm::ConfigurationDescriptions& );

  private:
    void produce( edm::Event&, const edm::EventSetup& ) override;

    edm::EDGetTokenT<edm::DetSetVector<TotemTimingRecHit> > recHitsToken_;
    int maxPlaneActiveChannels_;
    std::map<TotemTimingDetId,TotemTimingTrackRecognition> trk_algo_map_;
};

TotemTimingLocalTrackFitter::TotemTimingLocalTrackFitter( const edm::ParameterSet& iConfig ) :
  recHitsToken_( consumes<edm::DetSetVector<TotemTimingRecHit> >( iConfig.getParameter<edm::InputTag>( "recHitsTag" ) ) ),
  maxPlaneActiveChannels_( iConfig.getParameter<int>( "maxPlaneActiveChannels" ) )
{
  produces<edm::DetSetVector<TotemTimingLocalTrack> >();

  for ( unsigned short armNo = 0; armNo < 2; armNo++ )
    for ( unsigned short rpNo = 0; rpNo < 2; rpNo++ ) {
      TotemTimingDetId id( armNo, 1, rpNo, 0, 0 );
      TotemTimingTrackRecognition trk_algo( iConfig );
      trk_algo_map_.insert( std::make_pair( id, trk_algo ) );
    }
}

void
TotemTimingLocalTrackFitter::produce( edm::Event& iEvent, const edm::EventSetup& iSetup )
{
  std::unique_ptr<edm::DetSetVector<TotemTimingLocalTrack> > pOut( new edm::DetSetVector<TotemTimingLocalTrack> );

  edm::Handle<edm::DetSetVector<TotemTimingRecHit> > recHits;
  iEvent.getByToken( recHitsToken_, recHits );

  for ( const auto& trk_algo_entry : trk_algo_map_ )
    pOut->find_or_insert( trk_algo_entry.first );

  std::map<TotemTimingDetId,int> planeActivityMap;

  for ( const auto& vec: *recHits ) {
    const TotemTimingDetId detId( vec.detId() );
    TotemTimingDetId tmpId( 0, 1, 0, 0, 0 );
    tmpId.setArm( detId.arm() );
    tmpId.setRP( detId.rp() );
    tmpId.setPlane( detId.plane() );
    planeActivityMap[tmpId] += vec.size();
  }

  // feed hits to the track producers
  for ( const auto& vec : *recHits ) {
    const TotemTimingDetId detId( vec.detId() );

    TotemTimingDetId tmpId( 0, 1, 0, 0, 0 );
    tmpId.setArm( detId.arm() );
    tmpId.setRP( detId.rp() );
    tmpId.setPlane( detId.plane() );
    if ( planeActivityMap[tmpId] > maxPlaneActiveChannels_ )
      continue;

    for ( const auto& hit : vec ) {
      tmpId.setPlane( 0 );
      if ( trk_algo_map_.find( tmpId ) != trk_algo_map_.end() )
        trk_algo_map_.find( tmpId )->second.addHit( hit );
      else
        edm::LogWarning("TotemTimingLocalTrackFitter") << "Invalid detId for rechit: arm=" << detId.arm() << ", rp=" << detId.rp();
    }
  }

  // retrieves tracks for all hit sets
  for ( auto& trk_algo_entry : trk_algo_map_ )
    trk_algo_entry.second.produceTracks( pOut->operator[]( trk_algo_entry.first ) );

  iEvent.put( std::move( pOut ) );

  // remove all hits from the track producers to prepare for the next event
  for ( auto& trk_algo_entry : trk_algo_map_ )
    trk_algo_entry.second.clear();
}

void
TotemTimingLocalTrackFitter::fillDescriptions( edm::ConfigurationDescriptions& descr )
{
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>( "recHitsTag", edm::InputTag( "totemTimingRecHits" ) )
    ->setComment( "input rechits collection to retrieve" );
  desc.add<int>( "verbosity", 0 )
    ->setComment( "general verbosity of this module" );

  desc.add<double>( "threshold", 1.5 )
    ->setComment( "minimal number of rechits to be observed before launching the track recognition algorithm" );
  desc.add<double>( "thresholdFromMaximum", 0.5 )
    ->setComment( "threshold relative to hit profile function local maximum for determining the width of the track" );
  desc.add<double>( "resolution", 0.01 /* mm */ )
    ->setComment( "spatial resolution on the horizontal coordinate (in mm)" );
  desc.add<double>( "sigma", 0. )
    ->setComment( "pixel efficiency function parameter determining the smoothness of the step" );
  desc.add<double>( "tolerance", 0. /* mm */)
    ->setComment( "tolerance used for checking if the track contains certain hit" );
  desc.add<int>( "maxPlaneActiveChannels", 3 /* mm */)
    ->setComment( "threshold for discriminating noisy planes" );

  desc.add<std::string>( "pixelEfficiencyFunction", "(x>[0]-0.5*[1]-0.05)*(x<[0]+0.5*[1]-0.05)+0*[2]" )
    ->setComment( "efficiency function for single pixel\n"
                  "can be defined as:\n"
                  " * Precise: (TMath::Erf((x-[0]+0.5*([1]-0.05))/([2]/4)+2)+1)*TMath::Erfc((x-[0]-0.5*([1]-0.05))/([2]/4)-2)/4\n"
                  " * Fast: (x>[0]-0.5*([1]-0.05))*(x<[0]+0.5*([1]-0.05))+((x-[0]+0.5*([1]-0.05)+[2])/[2])*(x>[0]-0.5*([1]-0.05)-[2])*(x<[0]-0.5*([1]-0.05))+(2-(x-[0]-0.5*([1]-0.05)+[2])/[2])*(x>[0]+0.5*([1]-0.05))*(x<[0]+0.5*([1]-0.05)+[2])\n"
                  " * Legacy: (1/(1+exp(-(x-[0]+0.5*([1]-0.05))/[2])))*(1/(1+exp((x-[0]-0.5*([1]-0.05))/[2])))\n"
                  " * Default (sigma ignored): (x>[0]-0.5*[1]-0.05)*(x<[0]+0.5*[1]-0.05)+0*[2]\n"
                  "with:\n"
                  "  [0]: centre of pad\n"
                  "  [1]: width of pad\n"
                  "  [2]: sigma: distance between efficiency ~100 -> 0 outside width" );

  desc.add<double>( "yPosition", 0.0 )
    ->setComment( "vertical offset of the outcoming track centre" );
  desc.add<double>( "yWidth", 0.0 )
    ->setComment( "vertical track width" );

  descr.add( "totemTimingLocalTracks", desc );
}

DEFINE_FWK_MODULE( TotemTimingLocalTrackFitter );

