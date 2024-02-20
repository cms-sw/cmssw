#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"

#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/HGCalReco/interface/MtdHostCollection.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
//#include "DataFormats/TrackReco/interface/TrackBase.h"

using namespace edm;

class MTDSoAProducer : public edm::stream::EDProducer<> {
public:
  MTDSoAProducer(const ParameterSet& pset);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::Event& ev, const edm::EventSetup& es) final;

private:
  edm::EDGetTokenT<reco::TrackCollection> tracksToken_;
  edm::EDGetTokenT<edm::ValueMap<int>> trackAssocToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> t0Token_;
  edm::EDGetTokenT<edm::ValueMap<float>> sigmat0Token_;
  edm::EDGetTokenT<edm::ValueMap<float>> tmtdToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> sigmatmtdToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> betaToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> pathToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> MVAQualityToken_;
  edm::EDGetTokenT<edm::ValueMap<GlobalPoint>> posInMtdToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> momentumWithMTDToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> probPiToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> probKToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> probPToken_;
};

MTDSoAProducer::MTDSoAProducer(const ParameterSet& iConfig)
    : tracksToken_(consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("tracksSrc"))),
      trackAssocToken_(consumes<edm::ValueMap<int>>(iConfig.getParameter<edm::InputTag>("trackAssocSrc"))),
      t0Token_(consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("t0Src"))),
      sigmat0Token_(consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("sigmat0Src"))),
      tmtdToken_(consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("tmtdSrc"))),
      sigmatmtdToken_(consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("sigmatmtdSrc"))),
      betaToken_(consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("betamtd"))),
      pathToken_(consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("pathmtd"))),
      MVAQualityToken_(consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("mvaquality"))),
      posInMtdToken_(consumes<edm::ValueMap<GlobalPoint>>(iConfig.getParameter<edm::InputTag>("posmtd"))),
      momentumWithMTDToken_(consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("momentum"))),
      probPiToken_(consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("probPi"))),
      probKToken_(consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("probK"))),
      probPToken_(consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("probP"))) {
  produces<MtdHostCollection>();
}

// Configuration descriptions
void MTDSoAProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("tracksSrc", edm::InputTag("generalTracks"));
  desc.add<edm::InputTag>("trackAssocSrc", edm::InputTag("trackExtenderWithMTD:generalTrackassoc"));
  desc.add<edm::InputTag>("t0Src", edm::InputTag("tofPID:t0"));
  desc.add<edm::InputTag>("sigmat0Src", edm::InputTag("tofPID:sigmat0"));
  desc.add<edm::InputTag>("tmtdSrc", edm::InputTag("trackExtenderWithMTD:generalTracktmtd"));
  desc.add<edm::InputTag>("sigmatmtdSrc", edm::InputTag("trackExtenderWithMTD:generalTracksigmatmtd"));
  desc.add<edm::InputTag>("betamtd", edm::InputTag("trackExtenderWithMTD:generalTrackBeta"));
  desc.add<edm::InputTag>("pathmtd", edm::InputTag("trackExtenderWithMTD:generalTrackPathLength"));
  desc.add<edm::InputTag>("mvaquality", edm::InputTag("mtdTrackQualityMVA:mtdQualMVA"));
  desc.add<edm::InputTag>("posmtd", edm::InputTag("trackExtenderWithMTD:generalTrackmtdpos"));
  desc.add<edm::InputTag>("momentum", edm::InputTag("trackExtenderWithMTD:generalTrackp"));
  desc.add<edm::InputTag>("probPi", edm::InputTag("tofPID:probPi"));
  desc.add<edm::InputTag>("probK", edm::InputTag("tofPID:probK"));
  desc.add<edm::InputTag>("probP", edm::InputTag("tofPID:probP"));

  descriptions.add("mtdSoAProducer", desc);
}

void MTDSoAProducer::produce(edm::Event& ev, const edm::EventSetup& es) {
  edm::Handle<reco::TrackCollection> tracksH;
  ev.getByToken(tracksToken_, tracksH);
  const auto& tracks = *tracksH;

  const auto& trackAssoc = ev.get(trackAssocToken_);

  const auto& t0 = ev.get(t0Token_);
  const auto& sigmat0 = ev.get(sigmat0Token_);

  const auto& tmtd = ev.get(tmtdToken_);
  const auto& sigmatmtd = ev.get(sigmatmtdToken_);

  const auto& beta = ev.get(betaToken_);
  const auto& path = ev.get(pathToken_);
  const auto& MVAquality = ev.get(MVAQualityToken_);
  const auto& posInMTD = ev.get(posInMtdToken_);
  const auto& momentum = ev.get(momentumWithMTDToken_);
  const auto& probPi = ev.get(probPiToken_);
  const auto& probK = ev.get(probKToken_);
  const auto& probP = ev.get(probPToken_);

  auto MtdInfo = std::make_unique<MtdHostCollection>(tracks.size(), cms::alpakatools::host());

  auto& MtdInfoView = MtdInfo->view();
  for (unsigned int iTrack = 0; iTrack < tracks.size(); ++iTrack) {
    const reco::TrackRef trackref(tracksH, iTrack);

    if (trackAssoc[trackref] == -1) {
      MtdInfoView.trackAsocMTD()[iTrack] = -1;
      MtdInfoView.time0()[iTrack] = 0.f;
      MtdInfoView.time0Err()[iTrack] = -1.f;
      MtdInfoView.time()[iTrack] = 0.f;
      MtdInfoView.timeErr()[iTrack] = -1.f;
      MtdInfoView.MVAquality()[iTrack] = 0.f;
      MtdInfoView.pathLength()[iTrack] = 0.f;
      MtdInfoView.beta()[iTrack] = 0.f;
      MtdInfoView.posInMTD_x()[iTrack] = 0.f;
      MtdInfoView.posInMTD_y()[iTrack] = 0.f;
      MtdInfoView.posInMTD_z()[iTrack] = 0.f;
      MtdInfoView.momentumWithMTD()[iTrack] = 0.f;
      MtdInfoView.probPi()[iTrack] = 0.f;
      MtdInfoView.probK()[iTrack] = 0.f;
      MtdInfoView.probP()[iTrack] = 0.f;
      continue;
    }

    MtdInfoView.trackAsocMTD()[iTrack] = trackAssoc[trackref];
    MtdInfoView.time0()[iTrack] = t0[trackref];
    MtdInfoView.time0Err()[iTrack] = sigmat0[trackref];
    MtdInfoView.time()[iTrack] = tmtd[trackref];
    MtdInfoView.timeErr()[iTrack] = sigmatmtd[trackref];
    MtdInfoView.MVAquality()[iTrack] = MVAquality[trackref];
    MtdInfoView.pathLength()[iTrack] = path[trackref];
    MtdInfoView.beta()[iTrack] = beta[trackref];
    MtdInfoView.posInMTD_x()[iTrack] = posInMTD[trackref].x();
    MtdInfoView.posInMTD_y()[iTrack] = posInMTD[trackref].y();
    MtdInfoView.posInMTD_z()[iTrack] = posInMTD[trackref].z();
    MtdInfoView.momentumWithMTD()[iTrack] = momentum[trackref];
    MtdInfoView.probPi()[iTrack] = probPi[trackref];
    MtdInfoView.probK()[iTrack] = probK[trackref];
    MtdInfoView.probP()[iTrack] = probP[trackref];
  }

  ev.put(std::move(MtdInfo));
}

//define this as a plug-in
#include <FWCore/Framework/interface/MakerMacros.h>
DEFINE_FWK_MODULE(MTDSoAProducer);
