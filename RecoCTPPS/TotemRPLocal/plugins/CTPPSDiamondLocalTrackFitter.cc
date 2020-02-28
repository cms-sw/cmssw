/****************************************************************************
 *
 * This is a part of PPS offline software.
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

class CTPPSDiamondLocalTrackFitter : public edm::stream::EDProducer<> {
public:
  explicit CTPPSDiamondLocalTrackFitter(const edm::ParameterSet&);
  ~CTPPSDiamondLocalTrackFitter() override;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  edm::EDGetTokenT<edm::DetSetVector<CTPPSDiamondRecHit> > recHitsToken_;
  CTPPSDiamondTrackRecognition trk_algo_45_;
  CTPPSDiamondTrackRecognition trk_algo_56_;
};

CTPPSDiamondLocalTrackFitter::CTPPSDiamondLocalTrackFitter(const edm::ParameterSet& iConfig)
    : recHitsToken_(
          consumes<edm::DetSetVector<CTPPSDiamondRecHit> >(iConfig.getParameter<edm::InputTag>("recHitsTag"))),
      trk_algo_45_(iConfig.getParameter<edm::ParameterSet>("trackingAlgorithmParams")),
      trk_algo_56_(iConfig.getParameter<edm::ParameterSet>("trackingAlgorithmParams")) {
  produces<edm::DetSetVector<CTPPSDiamondLocalTrack> >();
}

CTPPSDiamondLocalTrackFitter::~CTPPSDiamondLocalTrackFitter() {}

void CTPPSDiamondLocalTrackFitter::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto pOut = std::make_unique<edm::DetSetVector<CTPPSDiamondLocalTrack> >();

  edm::Handle<edm::DetSetVector<CTPPSDiamondRecHit> > recHits;
  iEvent.getByToken(recHitsToken_, recHits);

  const CTPPSDiamondDetId id_45(0, 1, 6, 0, 0), id_56(1, 1, 6, 0, 0);

  pOut->find_or_insert(id_45);                                                  // tracks in 4-5
  edm::DetSet<CTPPSDiamondLocalTrack>& tracks56 = pOut->find_or_insert(id_56);  // tracks in 5-6

  // workaround to retrieve the detset for 4-5 without losing the reference
  edm::DetSet<CTPPSDiamondLocalTrack>& tracks45 = pOut->operator[](id_45);

  // feed hits to the track producers
  for (const auto& vec : *recHits) {
    const CTPPSDiamondDetId detid(vec.detId());
    for (const auto& hit : vec) {
      // skip hits without a leading edge
      if (hit.ootIndex() == CTPPSDiamondRecHit::TIMESLICE_WITHOUT_LEADING)
        continue;

      switch (detid.arm()) {
        case 0: {
          trk_algo_45_.addHit(hit);
        } break;
        case 1: {
          trk_algo_56_.addHit(hit);
        } break;
        default:
          edm::LogWarning("CTPPSDiamondLocalTrackFitter") << "Invalid arm for rechit: " << detid.arm();
          break;
      }
    }
  }

  // retrieve the tracks for both arms
  trk_algo_45_.produceTracks(tracks45);
  trk_algo_56_.produceTracks(tracks56);

  iEvent.put(std::move(pOut));

  // remove all hits from the track producers to prepare for the next event
  trk_algo_45_.clear();
  trk_algo_56_.clear();
}

void CTPPSDiamondLocalTrackFitter::fillDescriptions(edm::ConfigurationDescriptions& descr) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("recHitsTag", edm::InputTag("ctppsDiamondRecHits"))
      ->setComment("input rechits collection to retrieve");

  edm::ParameterSetDescription trackingAlgoParams;
  trackingAlgoParams.add<double>("threshold", 1.5)
      ->setComment("minimal number of rechits to be observed before launching the track recognition algorithm");
  trackingAlgoParams.add<double>("thresholdFromMaximum", 0.5);
  trackingAlgoParams.add<double>("resolution", 0.01 /* mm */)
      ->setComment("spatial resolution on the horizontal coordinate (in mm)");
  trackingAlgoParams.add<double>("sigma", 0.1);
  trackingAlgoParams.add<double>("startFromX", -0.5 /* mm */)
      ->setComment("starting horizontal coordinate of rechits for the track recognition");
  trackingAlgoParams.add<double>("stopAtX", 19.5 /* mm */)
      ->setComment("ending horizontal coordinate of rechits for the track recognition");
  trackingAlgoParams.add<double>("tolerance", 0.1 /* mm */)
      ->setComment("tolerance used for checking if the track contains certain hit");

  trackingAlgoParams.add<std::string>("pixelEfficiencyFunction", "(x>[0]-0.5*[1])*(x<[0]+0.5*[1])+0*[2]")
      ->setComment(
          "efficiency function for single pixel\n"
          "can be defined as:\n"
          " * Precise: (TMath::Erf((x-[0]+0.5*[1])/([2]/4)+2)+1)*TMath::Erfc((x-[0]-0.5*[1])/([2]/4)-2)/4\n"
          " * Fast: "
          "(x>[0]-0.5*[1])*(x<[0]+0.5*[1])+((x-[0]+0.5*[1]+[2])/"
          "[2])*(x>[0]-0.5*[1]-[2])*(x<[0]-0.5*[1])+(2-(x-[0]-0.5*[1]+[2])/[2])*(x>[0]+0.5*[1])*(x<[0]+0.5*[1]+[2])\n"
          " * Legacy: (1/(1+exp(-(x-[0]+0.5*[1])/[2])))*(1/(1+exp((x-[0]-0.5*[1])/[2])))\n"
          " * Default (sigma ignored): (x>[0]-0.5*[1])*(x<[0]+0.5*[1])+0*[2]\n"
          "with:\n"
          "  [0]: centre of pad\n"
          "  [1]: width of pad\n"
          "  [2]: sigma: distance between efficiency ~100 -> 0 outside width");

  trackingAlgoParams.add<double>("yPosition", 0.0)->setComment("vertical offset of the outcoming track centre");
  trackingAlgoParams.add<double>("yWidth", 0.0)->setComment("vertical track width");
  trackingAlgoParams.add<bool>("excludeSingleEdgeHits", true)
      ->setComment("exclude rechits with missing leading/trailing edge");

  desc.add<edm::ParameterSetDescription>("trackingAlgorithmParams", trackingAlgoParams)
      ->setComment("list of parameters associated to the track recognition algorithm");

  descr.add("ctppsDiamondLocalTracks", desc);
}

DEFINE_FWK_MODULE(CTPPSDiamondLocalTrackFitter);
