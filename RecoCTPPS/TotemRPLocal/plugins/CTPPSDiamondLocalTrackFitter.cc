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
    CTPPSDiamondTrackRecognition* trk_algo_45_;
    CTPPSDiamondTrackRecognition* trk_algo_56_;
};

CTPPSDiamondLocalTrackFitter::CTPPSDiamondLocalTrackFitter( const edm::ParameterSet& iConfig ) :
  recHitsToken_( consumes< edm::DetSetVector<CTPPSDiamondRecHit> >( iConfig.getParameter<edm::InputTag>( "recHitsTag" ) ) ),
  trk_algo_45_ ( new CTPPSDiamondTrackRecognition( iConfig.getParameter<edm::ParameterSet>( "trackingAlgorithmParams" ) ) ),
  trk_algo_56_ ( new CTPPSDiamondTrackRecognition( iConfig.getParameter<edm::ParameterSet>( "trackingAlgorithmParams" ) ) )
{
  produces< edm::DetSetVector<CTPPSDiamondLocalTrack> >();
}

CTPPSDiamondLocalTrackFitter::~CTPPSDiamondLocalTrackFitter()
{
  if ( trk_algo_45_ ) delete trk_algo_45_;
  if ( trk_algo_56_ ) delete trk_algo_56_;
}

void
CTPPSDiamondLocalTrackFitter::produce( edm::Event& iEvent, const edm::EventSetup& iSetup )
{
  std::unique_ptr< edm::DetSetVector<CTPPSDiamondLocalTrack> > pOut( new edm::DetSetVector<CTPPSDiamondLocalTrack> );

  edm::Handle< edm::DetSetVector<CTPPSDiamondRecHit> > recHits;
  iEvent.getByToken( recHitsToken_, recHits );

  CTPPSDiamondDetId tempid_45(0,1,6,0,0);
  pOut->find_or_insert( tempid_45 );
  CTPPSDiamondDetId tempid_56(1,1,6,0,0);
  edm::DetSet<CTPPSDiamondLocalTrack>& tracks56 =pOut->find_or_insert( tempid_56 );
  edm::DetSet<CTPPSDiamondLocalTrack>& tracks45 = pOut->operator[]( tempid_45 );
  
  trk_algo_45_->clear();
  trk_algo_56_->clear();

  for ( edm::DetSetVector<CTPPSDiamondRecHit>::const_iterator vec = recHits->begin(); vec != recHits->end(); ++vec )
  {
    const CTPPSDiamondDetId detid( vec->detId() );

    if (detid.arm()==0) 
    {
      for ( edm::DetSet<CTPPSDiamondRecHit>::const_iterator hit = vec->begin(); hit != vec->end(); ++hit )
      {
	trk_algo_45_->addHit( *hit );
      }
    } else 
    {
      for ( edm::DetSet<CTPPSDiamondRecHit>::const_iterator hit = vec->begin(); hit != vec->end(); ++hit )
      {
	trk_algo_56_->addHit( *hit );
      }
    }
  }

  trk_algo_45_->produceTracks( tracks45 );
  trk_algo_56_->produceTracks( tracks56 );
  
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
