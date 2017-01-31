/****************************************************************************
 *
 * This is a part of CTPPS offline software.
 * Authors:
 *   Laurent Forthomme (laurent.forthomme@cern.ch)
 *   Nicola Minafra (nicola.minafra@cern.ch)
 *
 ****************************************************************************/

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/Common/interface/DetSetVector.h"

#include "DataFormats/CTPPSDetId/interface/CTPPSDiamondDetId.h"
#include "DataFormats/CTPPSReco/interface/CTPPSDiamondRecHit.h"
#include "DataFormats/CTPPSReco/interface/CTPPSDiamondLocalTrack.h"

#include "RecoCTPPS/TotemRPLocal/interface/CTPPSDiamondTrackRecognition.h"

class CTPPSDiamondLocalTrackFitter : public edm::stream::EDProducer<>
{
  public:
    explicit CTPPSDiamondLocalTrackFitter( const edm::ParameterSet& );
    ~CTPPSDiamondLocalTrackFitter();

    static void fillDescriptions( edm::ConfigurationDescriptions& );

  private:
    virtual void produce( edm::Event&, const edm::EventSetup& ) override;

    edm::EDGetTokenT< edm::DetSetVector<CTPPSDiamondRecHit> > recHitsToken_;
    CTPPSDiamondTrackRecognition* trk_algo_;
};

CTPPSDiamondLocalTrackFitter::CTPPSDiamondLocalTrackFitter( const edm::ParameterSet& iConfig ) :
  recHitsToken_( consumes< edm::DetSetVector<CTPPSDiamondRecHit> >( iConfig.getParameter<edm::InputTag>( "recHitsTag" ) ) ),
  trk_algo_ ( new CTPPSDiamondTrackRecognition( iConfig.getParameter<edm::ParameterSet>( "trackingAlgorithmParams" ) ) )
{
  produces< edm::DetSetVector<CTPPSDiamondLocalTrack> >();
}

CTPPSDiamondLocalTrackFitter::~CTPPSDiamondLocalTrackFitter()
{
  if ( trk_algo_ ) delete trk_algo_;
}

void
CTPPSDiamondLocalTrackFitter::produce( edm::Event& iEvent, const edm::EventSetup& iSetup )
{
  std::unique_ptr< edm::DetSetVector<CTPPSDiamondLocalTrack> > pOut( new edm::DetSetVector<CTPPSDiamondLocalTrack> );

  edm::Handle< edm::DetSetVector<CTPPSDiamondRecHit> > recHits;
  iEvent.getByToken( recHitsToken_, recHits );

  trk_algo_->clear();
  for ( edm::DetSetVector<CTPPSDiamondRecHit>::const_iterator vec = recHits->begin(); vec != recHits->end(); ++vec )
  {
    const CTPPSDiamondDetId detid( vec->detId() );

    edm::DetSet<CTPPSDiamondLocalTrack>& tracks = pOut->find_or_insert( detid );

    for ( edm::DetSet<CTPPSDiamondRecHit>::const_iterator hit = vec->begin(); hit != vec->end(); ++hit )
    {
      trk_algo_->addHit( *hit );
    }
    trk_algo_->produceTracks( tracks );
  }

  iEvent.put( std::move( pOut ) );
}

void
CTPPSDiamondLocalTrackFitter::fillDescriptions( edm::ConfigurationDescriptions& descr )
{
  // The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descr.addDefault( desc );
}

DEFINE_FWK_MODULE( CTPPSDiamondLocalTrackFitter );
