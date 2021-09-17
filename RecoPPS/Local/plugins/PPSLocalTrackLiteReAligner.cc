/****************************************************************************
 *
 * This is a part of TOTEM offline software.
 * Authors:
 *   Jan Kašpar (jan.kaspar@gmail.com)
 *
 ****************************************************************************/

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/CTPPSReco/interface/CTPPSLocalTrackLite.h"
#include "DataFormats/CTPPSReco/interface/CTPPSLocalTrackLiteFwd.h"

#include "CondFormats/AlignmentRecord/interface/RPRealAlignmentRecord.h"
#include "CondFormats/PPSObjects/interface/CTPPSRPAlignmentCorrectionsData.h"

//----------------------------------------------------------------------------------------------------

class PPSLocalTrackLiteReAligner : public edm::global::EDProducer<> {
public:
  explicit PPSLocalTrackLiteReAligner(const edm::ParameterSet &);

  void produce(edm::StreamID, edm::Event &, const edm::EventSetup &) const override;

  static void fillDescriptions(edm::ConfigurationDescriptions &);

private:
  const edm::EDGetTokenT<CTPPSLocalTrackLiteCollection> inputTrackToken_;

  const edm::ESGetToken<CTPPSRPAlignmentCorrectionsData, RPRealAlignmentRecord> alignmentToken_;

  const std::string outputTrackTag_;
};

//----------------------------------------------------------------------------------------------------

PPSLocalTrackLiteReAligner::PPSLocalTrackLiteReAligner(const edm::ParameterSet &iConfig)
    : inputTrackToken_(consumes<CTPPSLocalTrackLiteCollection>(iConfig.getParameter<edm::InputTag>("inputTrackTag"))),
      alignmentToken_(esConsumes<CTPPSRPAlignmentCorrectionsData, RPRealAlignmentRecord>(
          iConfig.getParameter<edm::ESInputTag>("alignmentTag"))),
      outputTrackTag_(iConfig.getParameter<std::string>("outputTrackTag")) {
  produces<CTPPSLocalTrackLiteCollection>(outputTrackTag_);
}

//----------------------------------------------------------------------------------------------------

void PPSLocalTrackLiteReAligner::fillDescriptions(edm::ConfigurationDescriptions &descr) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("inputTrackTag", edm::InputTag("ctppsLocalTrackLiteProducer"))
      ->setComment("tag of the input CTPPSLocalTrackLiteCollection");

  desc.add<edm::ESInputTag>("alignmentTag", edm::ESInputTag(""))->setComment("tag of the alignment data");

  desc.add<std::string>("outputTrackTag", "")->setComment("tag of the output CTPPSLocalTrackLiteCollection");

  descr.add("ppsLocalTrackLiteReAligner", desc);
}

//----------------------------------------------------------------------------------------------------

void PPSLocalTrackLiteReAligner::produce(edm::StreamID, edm::Event &iEvent, const edm::EventSetup &iSetup) const {
  // get alignment corrections
  auto const &alignment = iSetup.getData(alignmentToken_);

  // prepare output
  auto output = std::make_unique<CTPPSLocalTrackLiteCollection>();

  // apply alignment corrections
  auto const &inputTracks = iEvent.get(inputTrackToken_);
  for (const auto &tr : inputTracks) {
    auto it = alignment.getRPMap().find(tr.rpId());
    if (it == alignment.getRPMap().end()) {
      edm::LogError("PPS") << "Cannot find alignment correction for RP " << tr.rpId() << ". The track will be skipped.";
    } else {
      output->emplace_back(tr.rpId(),
                           tr.x() + it->second.getShX(),
                           tr.xUnc(),
                           tr.y() + it->second.getShY(),
                           tr.yUnc(),
                           tr.tx(),
                           tr.txUnc(),
                           tr.ty(),
                           tr.tyUnc(),
                           tr.chiSquaredOverNDF(),
                           tr.pixelTrackRecoInfo(),
                           tr.numberOfPointsUsedForFit(),
                           tr.time(),
                           tr.timeUnc());
    }
  }

  // save output to event
  iEvent.put(std::move(output), outputTrackTag_);
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(PPSLocalTrackLiteReAligner);
