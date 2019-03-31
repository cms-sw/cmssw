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
#include "DataFormats/Math/interface/libminifloat.h"

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

          float roundedX0 = MiniFloatConverter::reduceMantissaToNbitsRounding<14>(trk.getX0());
          float roundedX0Sigma = MiniFloatConverter::reduceMantissaToNbitsRounding<8>(trk.getX0Sigma());
          float roundedY0 = MiniFloatConverter::reduceMantissaToNbitsRounding<13>(trk.getY0());
          float roundedY0Sigma = MiniFloatConverter::reduceMantissaToNbitsRounding<8>(trk.getY0Sigma());
          float roundedTx = MiniFloatConverter::reduceMantissaToNbitsRounding<11>(trk.getTx());
          float roundedTxSigma = MiniFloatConverter::reduceMantissaToNbitsRounding<8>(trk.getTxSigma());
          float roundedTy = MiniFloatConverter::reduceMantissaToNbitsRounding<11>(trk.getTy());
          float roundedTySigma = MiniFloatConverter::reduceMantissaToNbitsRounding<8>(trk.getTySigma());
          float roundedChiSquaredOverNDF = MiniFloatConverter::reduceMantissaToNbitsRounding<8>(trk.getChiSquaredOverNDF());

        pOut->emplace_back( rpId, roundedX0, roundedX0Sigma, roundedY0, roundedY0Sigma, roundedTx, roundedTxSigma, roundedTy, roundedTySigma, 
        roundedChiSquaredOverNDF, CTPPSpixelLocalTrackReconstructionInfo::invalid, trk.getNumberOfPointsUsedForFit(),0,0 );
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
        float roundedX0 = MiniFloatConverter::reduceMantissaToNbitsRounding<16>(trk.getX0());
        float roundedX0Sigma = MiniFloatConverter::reduceMantissaToNbitsRounding<8>(trk.getX0Sigma());
        float roundedY0 = MiniFloatConverter::reduceMantissaToNbitsRounding<13>(trk.getY0());
        float roundedY0Sigma = MiniFloatConverter::reduceMantissaToNbitsRounding<8>(trk.getY0Sigma());
        float roundedT = MiniFloatConverter::reduceMantissaToNbitsRounding<16>(trk.getT());
        float roundedTSigma = MiniFloatConverter::reduceMantissaToNbitsRounding<13>(trk.getTSigma());

        pOut->emplace_back( rpId, roundedX0, roundedX0Sigma, roundedY0, roundedY0Sigma, 0., 0., 0., 0., 0., 
        CTPPSpixelLocalTrackReconstructionInfo::invalid, trk.getNumOfPlanes(), roundedT, roundedTSigma);
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
             && trk.getTy()>pixelTrackTyMin_ && trk.getTy()<pixelTrackTyMax_){
            float roundedX0 = MiniFloatConverter::reduceMantissaToNbitsRounding<16>(trk.getX0());
            float roundedX0Sigma = MiniFloatConverter::reduceMantissaToNbitsRounding<8>(trk.getX0Sigma());
            float roundedY0 = MiniFloatConverter::reduceMantissaToNbitsRounding<13>(trk.getY0());
            float roundedY0Sigma = MiniFloatConverter::reduceMantissaToNbitsRounding<8>(trk.getY0Sigma());
            float roundedTx = MiniFloatConverter::reduceMantissaToNbitsRounding<11>(trk.getTx());
            float roundedTxSigma = MiniFloatConverter::reduceMantissaToNbitsRounding<8>(trk.getTxSigma());
            float roundedTy = MiniFloatConverter::reduceMantissaToNbitsRounding<11>(trk.getTy());
            float roundedTySigma = MiniFloatConverter::reduceMantissaToNbitsRounding<8>(trk.getTySigma());
            float roundedChiSquaredOverNDF = MiniFloatConverter::reduceMantissaToNbitsRounding<8>(trk.getChiSquaredOverNDF());

            pOut->emplace_back( rpId, roundedX0, roundedX0Sigma, roundedY0, roundedY0Sigma, roundedTx, roundedTxSigma, roundedTy, roundedTySigma, 
            roundedChiSquaredOverNDF, trk.getRecoInfo(), trk.getNumberOfPointsUsedForFit(),0.,0. );
          }
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

  // By default: module enabled (doNothing=false), but all includeXYZ flags set to false.
  // The includeXYZ are switched on when the "ctpps_2016" era is declared in python config, see:
  // RecoCTPPS/TotemRPLocal/python/ctppsLocalTrackLiteProducer_cff.py

  desc.add<bool>("includeStrips", false)->setComment("whether tracks from Si strips should be included");
  desc.add<edm::InputTag>( "tagSiStripTrack", edm::InputTag( "totemRPLocalTrackFitter" ) )
    ->setComment( "input TOTEM strips' local tracks collection to retrieve" );

  desc.add<bool>("includeDiamonds", false)->setComment("whether tracks from diamonds strips should be included");
  desc.add<edm::InputTag>( "tagDiamondTrack", edm::InputTag( "ctppsDiamondLocalTracks" ) )
    ->setComment( "input diamond detectors' local tracks collection to retrieve" );

  desc.add<bool>("includePixels", false)->setComment("whether tracks from pixels should be included");
  desc.add<edm::InputTag>( "tagPixelTrack", edm::InputTag( "ctppsPixelLocalTracks"   ) )
    ->setComment( "input pixel detectors' local tracks collection to retrieve" );

  desc.add<bool>("doNothing", false)
    ->setComment("disable the module");

  desc.add<double>("pixelTrackTxMin",-10.0);
  desc.add<double>("pixelTrackTxMax", 10.0);
  desc.add<double>("pixelTrackTyMin",-10.0);
  desc.add<double>("pixelTrackTyMax", 10.0);

  descr.add( "ctppsLocalTrackLiteDefaultProducer", desc );
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE( CTPPSLocalTrackLiteProducer );
