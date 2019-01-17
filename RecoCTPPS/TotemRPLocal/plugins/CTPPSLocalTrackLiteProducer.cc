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

  double pixelTrackTxMin_,pixelTrackTxMax_,pixelTrackTyMin_,pixelTrackTyMax_;
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

  pixelTrackTxMin_ = iConfig.getParameter<double>("pixelTrackTxMin");
  pixelTrackTxMax_ = iConfig.getParameter<double>("pixelTrackTxMax");
  pixelTrackTyMin_ = iConfig.getParameter<double>("pixelTrackTyMin");
  pixelTrackTyMax_ = iConfig.getParameter<double>("pixelTrackTyMax");
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
        pOut->emplace_back( rpId, trk.getX0(), trk.getX0Sigma(), trk.getY0(), trk.getY0Sigma(), trk.getTx(), trk.getTxSigma(), trk.getTy(), trk.getTySigma(), 
        trk.getChiSquaredOverNDF(), CTPPSReconstructionInfo::invalid, trk.getNumberOfPointsUsedForFit(),0,0 );
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
        pOut->emplace_back( rpId, trk.getX0(), trk.getX0Sigma(), trk.getY0(), trk.getY0Sigma(), 
        0., 0., 0., 0., 0., CTPPSReconstructionInfo::invalid, trk.getNumOfPlanes(), trk.getT(), trk.getTSigma() );
      }
    }
  }


  //----- pixel detectors

  if (includePixels_)
  {
    edm::Handle< edm::DetSetVector<CTPPSPixelLocalTrack> > inputPixelTracks;
    if (not pixelTrackToken_.isUninitialized()){
      iEvent.getByToken( pixelTrackToken_, inputPixelTracks );

    // process tracks from pixels
      for ( const auto& rpv : *inputPixelTracks ) {
        const uint32_t rpId = rpv.detId();
        for ( const auto& trk : rpv ) {
      if ( !trk.isValid() ) continue;
      if(trk.getTx()>pixelTrackTxMin_ && trk.getTx()<pixelTrackTxMax_
         && trk.getTy()>pixelTrackTyMin_ && trk.getTy()<pixelTrackTyMax_)

        pOut->emplace_back( rpId, trk.getX0(), trk.getX0Sigma(), trk.getY0(), trk.getY0Sigma(), trk.getTx(), trk.getTxSigma(), trk.getTy(), trk.getTySigma(), 
        trk.getChiSquaredOverNDF(), trk.getRecoInfo(), trk.getNumberOfPointsUsedForFit(),0.,0. );
      
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

  desc.add<double>("pixelTrackTxMin",-10.0);
  desc.add<double>("pixelTrackTxMax", 10.0);
  desc.add<double>("pixelTrackTyMin",-10.0);
  desc.add<double>("pixelTrackTyMax", 10.0);

  descr.add( "ctppsLocalTrackLiteDefaultProducer", desc );
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE( CTPPSLocalTrackLiteProducer );
