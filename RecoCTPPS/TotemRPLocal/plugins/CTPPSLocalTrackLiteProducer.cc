/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors:
*   Jan Kašpar (jan.kaspar@gmail.com)
*   Laurent Forthomme
*
****************************************************************************/

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/CTPPSReco/interface/TotemRPLocalTrack.h"
#include "DataFormats/CTPPSReco/interface/CTPPSDiamondLocalTrack.h"

#include "DataFormats/CTPPSReco/interface/CTPPSLocalTrackLite.h"

//----------------------------------------------------------------------------------------------------

/**
 * \brief Distills the essential track data from all RPs.
 **/
class CTPPSLocalTrackLiteProducer : public edm::stream::EDProducer<>
{
  public:
    explicit CTPPSLocalTrackLiteProducer( const edm::ParameterSet& );
    virtual ~CTPPSLocalTrackLiteProducer() {}

    virtual void produce( edm::Event&, const edm::EventSetup& ) override;

  private:
    edm::EDGetTokenT< edm::DetSetVector<TotemRPLocalTrack> > siStripTrackToken_;
    edm::EDGetTokenT< edm::DetSetVector<CTPPSDiamondLocalTrack> > diamondTrackToken_;

    /// if true, this module will do nothing
    /// needed for consistency with CTPPS-less workflows
    bool doNothing_;
};

//----------------------------------------------------------------------------------------------------

CTPPSLocalTrackLiteProducer::CTPPSLocalTrackLiteProducer( const edm::ParameterSet& iConfig ) :
  siStripTrackToken_( consumes< edm::DetSetVector<TotemRPLocalTrack> >     ( iConfig.getParameter<edm::InputTag>("tagSiStripTrack") ) ),
  diamondTrackToken_( consumes< edm::DetSetVector<CTPPSDiamondLocalTrack> >( iConfig.getParameter<edm::InputTag>("tagDiamondTrack") ) ),
  doNothing_( iConfig.getParameter<bool>( "doNothing" ) )
{
  if ( doNothing_ )
    return;

  produces< std::vector<CTPPSLocalTrackLite> >( "TrackingStrip" );
  produces< std::vector<CTPPSLocalTrackLite> >( "TimingDiamond" );
}

//----------------------------------------------------------------------------------------------------
 
void
CTPPSLocalTrackLiteProducer::produce( edm::Event& iEvent, const edm::EventSetup& )
{
  if ( doNothing_ )
    return;

  //----- TOTEM strips

  // get input from Si strips
  edm::Handle< edm::DetSetVector<TotemRPLocalTrack> > inputSiStripTracks;
  iEvent.getByToken( siStripTrackToken_, inputSiStripTracks );

  // prepare output
  std::unique_ptr< std::vector<CTPPSLocalTrackLite> > pSiStripOut( new std::vector<CTPPSLocalTrackLite>() );
  
  // process tracks from Si strips
  for ( const auto rpv : *inputSiStripTracks ) {
    const uint32_t rpId = rpv.detId();
    for ( const auto t : rpv ) {
      if ( !t.isValid() ) continue;
      pSiStripOut->emplace_back( rpId, t.getX0(), t.getX0Sigma(), t.getY0(), t.getY0Sigma() );
    }
  }

  // save output to event
  iEvent.put( std::move( pSiStripOut ), "TrackingStrip" );

  //----- diamond detectors

  // get input from diamond detectors
  edm::Handle< edm::DetSetVector<CTPPSDiamondLocalTrack> > inputDiamondTracks;
  iEvent.getByToken( diamondTrackToken_, inputDiamondTracks );

  // prepare output
  std::unique_ptr< std::vector<CTPPSLocalTrackLite> > pDiamondOut( new std::vector<CTPPSLocalTrackLite>() );

  // process tracks from diamond detectors
  for ( const auto rpv : *inputDiamondTracks ) {
    const unsigned int rpId = rpv.detId();
    for ( const auto t : rpv ) {
      if ( !t.isValid() ) continue;
      pDiamondOut->emplace_back( rpId, t.getX0(), t.getX0Sigma(), t.getY0(), t.getY0Sigma(), t.getT() );
    }
  }

  // save output to event
  iEvent.put( std::move( pDiamondOut ), "TimingDiamond" );
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE( CTPPSLocalTrackLiteProducer );
