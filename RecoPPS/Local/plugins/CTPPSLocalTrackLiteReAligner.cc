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
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/CTPPSReco/interface/CTPPSLocalTrackLite.h"
#include "DataFormats/CTPPSReco/interface/CTPPSLocalTrackLiteFwd.h"

#include "CondFormats/AlignmentRecord/interface/RPRealAlignmentRecord.h"
#include "CondFormats/PPSObjects/interface/CTPPSRPAlignmentCorrectionsData.h"

//----------------------------------------------------------------------------------------------------

class CTPPSLocalTrackLiteReAligner : public edm::stream::EDProducer<> {
public:
  explicit CTPPSLocalTrackLiteReAligner(const edm::ParameterSet &);

  void produce(edm::Event &, const edm::EventSetup &) override;

  static void fillDescriptions(edm::ConfigurationDescriptions &);

private:
  edm::EDGetTokenT<CTPPSLocalTrackLiteCollection> inputTrackToken_;

  std::string alignmentLabel_;

  std::string outputTrackTag_;
};

//----------------------------------------------------------------------------------------------------

CTPPSLocalTrackLiteReAligner::CTPPSLocalTrackLiteReAligner(const edm::ParameterSet &iConfig)
    : inputTrackToken_(consumes<CTPPSLocalTrackLiteCollection>(iConfig.getParameter<edm::InputTag>("inputTrackTag"))),
      alignmentLabel_(iConfig.getParameter<std::string>("alignmentLabel")),
      outputTrackTag_(iConfig.getParameter<std::string>("outputTrackTag")) {
  produces<CTPPSLocalTrackLiteCollection>(outputTrackTag_);
}

//----------------------------------------------------------------------------------------------------

void CTPPSLocalTrackLiteReAligner::fillDescriptions(edm::ConfigurationDescriptions &descr) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("inputTrackTag", edm::InputTag("ctppsLocalTrackLiteProducer"))
      ->setComment("tag of the input CTPPSLocalTrackLiteCollection");

  desc.add<std::string>("alignmentLabel", "")->setComment("label of the alignment data");

  desc.add<std::string>("outputTrackTag", "")->setComment("tag of the output CTPPSLocalTrackLiteCollection");

  descr.add("ctppsLocalTrackLiteReAligner", desc);
}

//----------------------------------------------------------------------------------------------------

void CTPPSLocalTrackLiteReAligner::produce(edm::Event &iEvent, const edm::EventSetup &iSetup) {
  // get alignment correction
  edm::ESHandle<CTPPSRPAlignmentCorrectionsData> hAlignment;
  iSetup.get<RPRealAlignmentRecord>().get(alignmentLabel_, hAlignment);

  // get input
  edm::Handle<CTPPSLocalTrackLiteCollection> hInputTracks;
  iEvent.getByToken(inputTrackToken_, hInputTracks);

  // prepare output
  auto output = std::make_unique<CTPPSLocalTrackLiteCollection>();

  // apply alignment correction
  for (const auto &tr : *hInputTracks) {
    auto it = hAlignment->getRPMap().find(tr.rpId());
    if (it == hAlignment->getRPMap().end()) {
      edm::LogError("CTPPSLocalTrackLiteReAligner::produce")
          << "Cannot find alignment correction for RP " << tr.rpId() << ". The track will be skipped.";
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

DEFINE_FWK_MODULE(CTPPSLocalTrackLiteReAligner);
