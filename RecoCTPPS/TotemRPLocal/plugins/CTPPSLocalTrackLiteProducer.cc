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
  doNothing_( iConfig.getParameter<bool>( "doNothing" ) )
{
  if ( doNothing_ ) return;

  siStripTrackToken_ = consumes< edm::DetSetVector<TotemRPLocalTrack> >     ( iConfig.getParameter<edm::InputTag>("tagSiStripTrack") );
  diamondTrackToken_ = consumes< edm::DetSetVector<CTPPSDiamondLocalTrack> >( iConfig.getParameter<edm::InputTag>("tagDiamondTrack") );

  produces< std::vector<CTPPSLocalTrackLite> >();
}

//----------------------------------------------------------------------------------------------------
 
void
CTPPSLocalTrackLiteProducer::produce( edm::Event& iEvent, const edm::EventSetup& )
{
  if ( doNothing_ )
    return;

  // prepare output
  std::unique_ptr< std::vector<CTPPSLocalTrackLite> > pOut( new std::vector<CTPPSLocalTrackLite>() );
  
  //----- TOTEM strips

  // get input from Si strips
  edm::Handle< edm::DetSetVector<TotemRPLocalTrack> > inputSiStripTracks;
  iEvent.getByToken( siStripTrackToken_, inputSiStripTracks );

  // process tracks from Si strips
  for ( const auto& rpv : *inputSiStripTracks ) {
    const uint32_t rpId = rpv.detId();
    for ( const auto& trk : rpv ) {
      if ( !trk.isValid() ) continue;
      pOut->emplace_back( rpId, trk.getX0(), trk.getX0Sigma(), trk.getY0(), trk.getY0Sigma() );
    }
  }

  //----- diamond detectors

  // get input from diamond detectors
  edm::Handle< edm::DetSetVector<CTPPSDiamondLocalTrack> > inputDiamondTracks;
  iEvent.getByToken( diamondTrackToken_, inputDiamondTracks );

  // process tracks from diamond detectors
  for ( const auto& rpv : *inputDiamondTracks ) {
    const unsigned int rpId = rpv.detId();
    for ( const auto& trk : rpv ) {
      if ( !trk.isValid() ) continue;
      pOut->emplace_back( rpId, trk.getX0(), trk.getX0Sigma(), trk.getY0(), trk.getY0Sigma(), trk.getT() );
    }
  }

  // save output to event
  iEvent.put( std::move( pOut ) );
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE( CTPPSLocalTrackLiteProducer );
