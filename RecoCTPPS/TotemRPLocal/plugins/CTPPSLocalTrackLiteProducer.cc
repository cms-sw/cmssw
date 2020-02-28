/****************************************************************************
 *
 * This is a part of TOTEM offline software.
 * Authors:
 *   Jan Kašpar (jan.kaspar@gmail.com)
 *   Laurent Forthomme
 *
 ****************************************************************************/

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/CTPPSReco/interface/CTPPSDiamondLocalTrack.h"
#include "DataFormats/CTPPSReco/interface/CTPPSPixelLocalTrack.h"
#include "DataFormats/CTPPSReco/interface/TotemRPLocalTrack.h"
#include "DataFormats/Common/interface/DetSetVector.h"

#include "DataFormats/CTPPSReco/interface/CTPPSLocalTrackLite.h"
#include "DataFormats/CTPPSReco/interface/CTPPSLocalTrackLiteFwd.h"

#include "DataFormats/Math/interface/libminifloat.h"

//----------------------------------------------------------------------------------------------------

/**
 * \brief Distills the essential track data from all RPs.
 **/
class CTPPSLocalTrackLiteProducer : public edm::stream::EDProducer<> {
public:
  explicit CTPPSLocalTrackLiteProducer(const edm::ParameterSet &);

  void produce(edm::Event &, const edm::EventSetup &) override;
  static void fillDescriptions(edm::ConfigurationDescriptions &);

private:
  /// HPTDC time slice width, in ns
  static constexpr float HPTDC_TIME_SLICE_WIDTH = 25.;

  bool includeStrips_;
  edm::EDGetTokenT<edm::DetSetVector<TotemRPLocalTrack>> siStripTrackToken_;

  bool includeDiamonds_;
  edm::EDGetTokenT<edm::DetSetVector<CTPPSDiamondLocalTrack>> diamondTrackToken_;

  bool includePixels_;
  edm::EDGetTokenT<edm::DetSetVector<CTPPSPixelLocalTrack>> pixelTrackToken_;

  double pixelTrackTxMin_, pixelTrackTxMax_, pixelTrackTyMin_, pixelTrackTyMax_;
  double timingTrackTMin_, timingTrackTMax_;
};

//----------------------------------------------------------------------------------------------------

CTPPSLocalTrackLiteProducer::CTPPSLocalTrackLiteProducer(const edm::ParameterSet &iConfig)
    : includeStrips_(iConfig.getParameter<bool>("includeStrips")),
      includeDiamonds_(iConfig.getParameter<bool>("includeDiamonds")),
      includePixels_(iConfig.getParameter<bool>("includePixels")),
      pixelTrackTxMin_(iConfig.getParameter<double>("pixelTrackTxMin")),
      pixelTrackTxMax_(iConfig.getParameter<double>("pixelTrackTxMax")),
      pixelTrackTyMin_(iConfig.getParameter<double>("pixelTrackTyMin")),
      pixelTrackTyMax_(iConfig.getParameter<double>("pixelTrackTyMax")),
      timingTrackTMin_(iConfig.getParameter<double>("timingTrackTMin")),
      timingTrackTMax_(iConfig.getParameter<double>("timingTrackTMax")) {
  auto tagSiStripTrack = iConfig.getParameter<edm::InputTag>("tagSiStripTrack");
  if (!tagSiStripTrack.label().empty())
    siStripTrackToken_ = consumes<edm::DetSetVector<TotemRPLocalTrack>>(tagSiStripTrack);

  auto tagDiamondTrack = iConfig.getParameter<edm::InputTag>("tagDiamondTrack");
  if (!tagDiamondTrack.label().empty())
    diamondTrackToken_ = consumes<edm::DetSetVector<CTPPSDiamondLocalTrack>>(tagDiamondTrack);

  auto tagPixelTrack = iConfig.getParameter<edm::InputTag>("tagPixelTrack");
  if (!tagPixelTrack.label().empty())
    pixelTrackToken_ = consumes<edm::DetSetVector<CTPPSPixelLocalTrack>>(tagPixelTrack);

  produces<CTPPSLocalTrackLiteCollection>();
}

//----------------------------------------------------------------------------------------------------

void CTPPSLocalTrackLiteProducer::produce(edm::Event &iEvent, const edm::EventSetup &) {
  // prepare output
  auto pOut = std::make_unique<CTPPSLocalTrackLiteCollection>();

  //----- TOTEM strips

  // get input from Si strips
  if (includeStrips_) {
    edm::Handle<edm::DetSetVector<TotemRPLocalTrack>> inputSiStripTracks;
    iEvent.getByToken(siStripTrackToken_, inputSiStripTracks);

    // process tracks from Si strips
    for (const auto &rpv : *inputSiStripTracks) {
      const uint32_t rpId = rpv.detId();
      for (const auto &trk : rpv) {
        if (!trk.isValid())
          continue;

        float roundedX0 = MiniFloatConverter::reduceMantissaToNbitsRounding<14>(trk.x0());
        float roundedX0Sigma = MiniFloatConverter::reduceMantissaToNbitsRounding<8>(trk.x0Sigma());
        float roundedY0 = MiniFloatConverter::reduceMantissaToNbitsRounding<13>(trk.y0());
        float roundedY0Sigma = MiniFloatConverter::reduceMantissaToNbitsRounding<8>(trk.y0Sigma());
        float roundedTx = MiniFloatConverter::reduceMantissaToNbitsRounding<11>(trk.tx());
        float roundedTxSigma = MiniFloatConverter::reduceMantissaToNbitsRounding<8>(trk.txSigma());
        float roundedTy = MiniFloatConverter::reduceMantissaToNbitsRounding<11>(trk.ty());
        float roundedTySigma = MiniFloatConverter::reduceMantissaToNbitsRounding<8>(trk.tySigma());
        float roundedChiSquaredOverNDF = MiniFloatConverter::reduceMantissaToNbitsRounding<8>(trk.chiSquaredOverNDF());

        pOut->emplace_back(rpId,  // detector info
                                  // spatial info
                           roundedX0,
                           roundedX0Sigma,
                           roundedY0,
                           roundedY0Sigma,
                           // angular info
                           roundedTx,
                           roundedTxSigma,
                           roundedTy,
                           roundedTySigma,
                           // reconstruction info
                           roundedChiSquaredOverNDF,
                           CTPPSpixelLocalTrackReconstructionInfo::invalid,
                           trk.numberOfPointsUsedForFit(),
                           // timing info
                           0.,
                           0.);
      }
    }
  }

  //----- diamond detectors

  if (includeDiamonds_) {
    // get input from diamond detectors
    edm::Handle<edm::DetSetVector<CTPPSDiamondLocalTrack>> inputDiamondTracks;
    iEvent.getByToken(diamondTrackToken_, inputDiamondTracks);

    // process tracks from diamond detectors
    for (const auto &rpv : *inputDiamondTracks) {
      const unsigned int rpId = rpv.detId();
      for (const auto &trk : rpv) {
        if (!trk.isValid())
          continue;

        const float abs_time = trk.time() + trk.ootIndex() * HPTDC_TIME_SLICE_WIDTH;
        if (abs_time < timingTrackTMin_ || abs_time > timingTrackTMax_)
          continue;

        float roundedX0 = MiniFloatConverter::reduceMantissaToNbitsRounding<16>(trk.x0());
        float roundedX0Sigma = MiniFloatConverter::reduceMantissaToNbitsRounding<8>(trk.x0Sigma());
        float roundedY0 = MiniFloatConverter::reduceMantissaToNbitsRounding<13>(trk.y0());
        float roundedY0Sigma = MiniFloatConverter::reduceMantissaToNbitsRounding<8>(trk.y0Sigma());
        float roundedT = MiniFloatConverter::reduceMantissaToNbitsRounding<16>(abs_time);
        float roundedTSigma = MiniFloatConverter::reduceMantissaToNbitsRounding<13>(trk.timeSigma());

        pOut->emplace_back(rpId,  // detector info
                                  // spatial info
                           roundedX0,
                           roundedX0Sigma,
                           roundedY0,
                           roundedY0Sigma,
                           // angular info
                           0.,
                           0.,
                           0.,
                           0.,
                           // reconstruction info
                           0.,
                           CTPPSpixelLocalTrackReconstructionInfo::invalid,
                           trk.numberOfPlanes(),
                           // timing info
                           roundedT,
                           roundedTSigma);
      }
    }
  }

  //----- pixel detectors

  if (includePixels_) {
    edm::Handle<edm::DetSetVector<CTPPSPixelLocalTrack>> inputPixelTracks;
    if (!pixelTrackToken_.isUninitialized()) {
      iEvent.getByToken(pixelTrackToken_, inputPixelTracks);

      // process tracks from pixels
      for (const auto &rpv : *inputPixelTracks) {
        const uint32_t rpId = rpv.detId();
        for (const auto &trk : rpv) {
          if (!trk.isValid())
            continue;
          if (trk.tx() > pixelTrackTxMin_ && trk.tx() < pixelTrackTxMax_ && trk.ty() > pixelTrackTyMin_ &&
              trk.ty() < pixelTrackTyMax_) {
            float roundedX0 = MiniFloatConverter::reduceMantissaToNbitsRounding<16>(trk.x0());
            float roundedX0Sigma = MiniFloatConverter::reduceMantissaToNbitsRounding<8>(trk.x0Sigma());
            float roundedY0 = MiniFloatConverter::reduceMantissaToNbitsRounding<13>(trk.y0());
            float roundedY0Sigma = MiniFloatConverter::reduceMantissaToNbitsRounding<8>(trk.y0Sigma());
            float roundedTx = MiniFloatConverter::reduceMantissaToNbitsRounding<11>(trk.tx());
            float roundedTxSigma = MiniFloatConverter::reduceMantissaToNbitsRounding<8>(trk.txSigma());
            float roundedTy = MiniFloatConverter::reduceMantissaToNbitsRounding<11>(trk.ty());
            float roundedTySigma = MiniFloatConverter::reduceMantissaToNbitsRounding<8>(trk.tySigma());
            float roundedChiSquaredOverNDF =
                MiniFloatConverter::reduceMantissaToNbitsRounding<8>(trk.chiSquaredOverNDF());

            pOut->emplace_back(rpId,  // detector info
                                      // spatial info
                               roundedX0,
                               roundedX0Sigma,
                               roundedY0,
                               roundedY0Sigma,
                               // angular info
                               roundedTx,
                               roundedTxSigma,
                               roundedTy,
                               roundedTySigma,
                               // reconstruction info
                               roundedChiSquaredOverNDF,
                               trk.recoInfo(),
                               trk.numberOfPointsUsedForFit(),
                               // timing info
                               0.,
                               0.);
          }
        }
      }
    }
  }

  // save output to event
  iEvent.put(std::move(pOut));
}

//----------------------------------------------------------------------------------------------------

void CTPPSLocalTrackLiteProducer::fillDescriptions(edm::ConfigurationDescriptions &descr) {
  edm::ParameterSetDescription desc;

  // By default: all includeXYZ flags set to false.
  // The includeXYZ are switched on when the "ctpps_2016" era is declared in
  // python config, see:
  // RecoCTPPS/TotemRPLocal/python/ctppsLocalTrackLiteProducer_cff.py

  desc.add<bool>("includeStrips", false)->setComment("whether tracks from Si strips should be included");
  desc.add<edm::InputTag>("tagSiStripTrack", edm::InputTag("totemRPLocalTrackFitter"))
      ->setComment("input TOTEM strips' local tracks collection to retrieve");

  desc.add<bool>("includeDiamonds", false)->setComment("whether tracks from diamonds strips should be included");
  desc.add<edm::InputTag>("tagDiamondTrack", edm::InputTag("ctppsDiamondLocalTracks"))
      ->setComment("input diamond detectors' local tracks collection to retrieve");

  desc.add<bool>("includePixels", false)->setComment("whether tracks from pixels should be included");
  desc.add<edm::InputTag>("tagPixelTrack", edm::InputTag("ctppsPixelLocalTracks"))
      ->setComment("input pixel detectors' local tracks collection to retrieve");
  desc.add<double>("timingTrackTMin", -12.5)->setComment("minimal track time selection for timing detectors, in ns");
  desc.add<double>("timingTrackTMax", +12.5)->setComment("maximal track time selection for timing detectors, in ns");

  desc.add<double>("pixelTrackTxMin", -10.0);
  desc.add<double>("pixelTrackTxMax", 10.0);
  desc.add<double>("pixelTrackTyMin", -10.0);
  desc.add<double>("pixelTrackTyMax", 10.0);

  descr.add("ctppsLocalTrackLiteDefaultProducer", desc);
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(CTPPSLocalTrackLiteProducer);
