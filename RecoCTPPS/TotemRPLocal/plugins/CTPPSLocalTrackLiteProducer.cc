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
#include "DataFormats/CTPPSReco/interface/CTPPSPixelLocalTrack.h"

#include "DataFormats/CTPPSReco/interface/CTPPSLocalTrackLite.h"

//----------------------------------------------------------------------------------------------------

/**
 * \brief Distills the essential track data from all RPs.
 **/
class CTPPSLocalTrackLiteProducer : public edm::stream::EDProducer<>
{
public:
  explicit CTPPSLocalTrackLiteProducer( const edm::ParameterSet& );
  ~CTPPSLocalTrackLiteProducer() override {}

  void produce( edm::Event&, const edm::EventSetup& ) override;
  static void fillDescriptions( edm::ConfigurationDescriptions& );

private:
  bool includeStrips_;
  edm::EDGetTokenT< edm::DetSetVector<TotemRPLocalTrack> > siStripTrackToken_;

  bool includeDiamonds_;
  edm::EDGetTokenT< edm::DetSetVector<CTPPSDiamondLocalTrack> > diamondTrackToken_;

  bool includePixels_;
  edm::EDGetTokenT< edm::DetSetVector<CTPPSPixelLocalTrack> > pixelTrackToken_;

  std::vector<double> pixelTrackTxRange_;
  std::vector<double> pixelTrackTyRange_;
/// if true, this module will do nothing
/// needed for consistency with CTPPS-less workflows
  bool doNothing_;
};

//----------------------------------------------------------------------------------------------------

CTPPSLocalTrackLiteProducer::CTPPSLocalTrackLiteProducer( const edm::ParameterSet& iConfig ) :
  doNothing_( iConfig.getParameter<bool>( "doNothing" ) )
{
  if ( doNothing_ ) return;

  includeStrips_ = iConfig.getParameter<bool>("includeStrips");
  siStripTrackToken_ = consumes< edm::DetSetVector<TotemRPLocalTrack> >     ( iConfig.getParameter<edm::InputTag>("tagSiStripTrack") );

  includeDiamonds_ = iConfig.getParameter<bool>("includeDiamonds");
  diamondTrackToken_ = consumes< edm::DetSetVector<CTPPSDiamondLocalTrack> >( iConfig.getParameter<edm::InputTag>("tagDiamondTrack") );

  includePixels_ = iConfig.getParameter<bool>("includePixels");
  auto tagPixelTrack = iConfig.getParameter<edm::InputTag>("tagPixelTrack"); 
  if (not tagPixelTrack.label().empty()){
    pixelTrackToken_   = consumes< edm::DetSetVector<CTPPSPixelLocalTrack> >  (tagPixelTrack);
  }

  pixelTrackTxRange_ = iConfig.getParameter<std::vector<double> >("pixelTrackTxRange");
  pixelTrackTyRange_ = iConfig.getParameter<std::vector<double> >("pixelTrackTyRange");
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
  if (includeStrips_)
  {
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
  }

  //----- diamond detectors

  if (includeDiamonds_)
  {
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
  }


  //----- pixel detectors

  if (includePixels_)
  {
    // get input from pixel detectors
    if(pixelTrackTxRange_.size() != 2 || pixelTrackTyRange_.size() != 2) throw cms::Exception("CTPPSLocalTrackLiteProducer") 
                                 << "Wrong number of parameters in pixel track Tx/Ty range";
    edm::Handle< edm::DetSetVector<CTPPSPixelLocalTrack> > inputPixelTracks;
    if (not pixelTrackToken_.isUninitialized()){
      iEvent.getByToken( pixelTrackToken_, inputPixelTracks );

    // process tracks from pixels
      for ( const auto& rpv : *inputPixelTracks ) {
        const uint32_t rpId = rpv.detId();
        for ( const auto& trk : rpv ) {
      if ( !trk.isValid() ) continue;
      if(trk.getTx()>pixelTrackTxRange_.at(0) && trk.getTx()<pixelTrackTxRange_.at(1)
         && trk.getTy()>pixelTrackTyRange_.at(0) && trk.getTy()<pixelTrackTyRange_.at(1) )
        pOut->emplace_back( rpId, trk.getX0(), trk.getX0Sigma(), trk.getY0(), trk.getY0Sigma() );
        }
      }
    }
  }

  // save output to event
  iEvent.put( std::move( pOut ) );
}

//----------------------------------------------------------------------------------------------------

void
CTPPSLocalTrackLiteProducer::fillDescriptions( edm::ConfigurationDescriptions& descr )
{
  edm::ParameterSetDescription desc;

  desc.add<bool>("includeStrips", true)->setComment("whether tracks from Si strips should be included");
  desc.add<edm::InputTag>( "tagSiStripTrack", edm::InputTag( "totemRPLocalTrackFitter" ) )
    ->setComment( "input TOTEM strips' local tracks collection to retrieve" );

  desc.add<bool>("includeDiamonds", true)->setComment("whether tracks from diamonds strips should be included");
  desc.add<edm::InputTag>( "tagDiamondTrack", edm::InputTag( "ctppsDiamondLocalTracks" ) )
    ->setComment( "input diamond detectors' local tracks collection to retrieve" );

  desc.add<bool>("includePixels", true)->setComment("whether tracks from pixels should be included");
  desc.add<edm::InputTag>( "tagPixelTrack"  , edm::InputTag( "ctppsPixelLocalTracks"   ) )
    ->setComment( "input pixel detectors' local tracks collection to retrieve" );
  desc.add<bool>( "doNothing", true ) // disable the module by default
    ->setComment( "disable the module" );

  desc.add<std::vector<double> >("pixelTrackTxRange",std::vector<double>({-0.03,0.03}) );
  desc.add<std::vector<double> >("pixelTrackTyRange",std::vector<double>({-0.04,0.04}) );

  descr.add( "ctppsLocalTrackLiteDefaultProducer", desc );
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE( CTPPSLocalTrackLiteProducer );
