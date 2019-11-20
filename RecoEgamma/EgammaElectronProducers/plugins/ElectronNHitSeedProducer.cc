//******************************************************************************
//
// Part of the refactorisation of of the E/gamma pixel matching for 2017 pixels
// This refactorisation converts the monolithic  approach to a series of
// independent producer modules, with each modules performing  a specific
// job as recommended by the 2017 tracker framework
//
//
// The module produces the ElectronSeeds, similarly to ElectronSeedProducer
// although with a varible number of required hits
//
//
// Author : Sam Harper (RAL), 2017
//
//*******************************************************************************

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "RecoEgamma/EgammaElectronAlgos/interface/TrajSeedMatcher.h"

class ElectronNHitSeedProducer : public edm::stream::EDProducer<> {
public:
  explicit ElectronNHitSeedProducer(const edm::ParameterSet&);
  ~ElectronNHitSeedProducer() override = default;

  void produce(edm::Event&, const edm::EventSetup&) final;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  TrajSeedMatcher matcher_;

  std::vector<edm::EDGetTokenT<std::vector<reco::SuperClusterRef>>> superClustersTokens_;
  edm::EDGetTokenT<TrajectorySeedCollection> initialSeedsToken_;
  edm::EDGetTokenT<std::vector<reco::Vertex>> verticesToken_;
  edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;
  edm::EDGetTokenT<MeasurementTrackerEvent> measTkEvtToken_;
};

namespace {
  template <typename T>
  edm::Handle<T> getHandle(const edm::Event& event, const edm::EDGetTokenT<T>& token) {
    edm::Handle<T> handle;
    event.getByToken(token, handle);
    return handle;
  }

  template <typename T>
  GlobalPoint convertToGP(const T& orgPoint) {
    return GlobalPoint(orgPoint.x(), orgPoint.y(), orgPoint.z());
  }

  int getLayerOrDiskNr(DetId detId, const TrackerTopology& trackerTopo) {
    if (detId.subdetId() == PixelSubdetector::PixelBarrel) {
      return trackerTopo.pxbLayer(detId);
    } else if (detId.subdetId() == PixelSubdetector::PixelEndcap) {
      return trackerTopo.pxfDisk(detId);
    } else
      return -1;
  }

  reco::ElectronSeed::PMVars makeSeedPixelVar(const TrajSeedMatcher::MatchInfo& matchInfo,
                                              const TrackerTopology& trackerTopo) {
    int layerOrDisk = getLayerOrDiskNr(matchInfo.detId, trackerTopo);
    reco::ElectronSeed::PMVars pmVars;
    pmVars.setDet(matchInfo.detId, layerOrDisk);
    pmVars.setDPhi(matchInfo.dPhiPos, matchInfo.dPhiNeg);
    pmVars.setDRZ(matchInfo.dRZPos, matchInfo.dRZNeg);

    return pmVars;
  }

}  // namespace

ElectronNHitSeedProducer::ElectronNHitSeedProducer(const edm::ParameterSet& pset)
    : matcher_(pset.getParameter<edm::ParameterSet>("matcherConfig")),
      initialSeedsToken_(consumes<TrajectorySeedCollection>(pset.getParameter<edm::InputTag>("initialSeeds"))),
      verticesToken_(consumes<std::vector<reco::Vertex>>(pset.getParameter<edm::InputTag>("vertices"))),
      beamSpotToken_(consumes<reco::BeamSpot>(pset.getParameter<edm::InputTag>("beamSpot"))),
      measTkEvtToken_(consumes<MeasurementTrackerEvent>(pset.getParameter<edm::InputTag>("measTkEvt"))) {
  const auto superClusTags = pset.getParameter<std::vector<edm::InputTag>>("superClusters");
  for (const auto& scTag : superClusTags) {
    superClustersTokens_.emplace_back(consumes<std::vector<reco::SuperClusterRef>>(scTag));
  }
  produces<reco::ElectronSeedCollection>();
}

void ElectronNHitSeedProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("initialSeeds", edm::InputTag("hltElePixelSeedsCombined"));
  desc.add<edm::InputTag>("vertices", edm::InputTag());
  desc.add<edm::InputTag>("beamSpot", edm::InputTag("hltOnlineBeamSpot"));
  desc.add<edm::InputTag>("measTkEvt", edm::InputTag("hltSiStripClusters"));
  desc.add<std::vector<edm::InputTag>>("superClusters",
                                       std::vector<edm::InputTag>{edm::InputTag{"hltEgammaSuperClustersToPixelMatch"}});
  desc.add<edm::ParameterSetDescription>("matcherConfig", TrajSeedMatcher::makePSetDescription());

  descriptions.add("electronNHitSeedProducer", desc);
}

void ElectronNHitSeedProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::ESHandle<TrackerTopology> trackerTopoHandle;
  iSetup.get<TrackerTopologyRcd>().get(trackerTopoHandle);

  matcher_.doEventSetup(iSetup);
  matcher_.setMeasTkEvtHandle(getHandle(iEvent, measTkEvtToken_));

  auto eleSeeds = std::make_unique<reco::ElectronSeedCollection>();
  auto initialSeedsHandle = getHandle(iEvent, initialSeedsToken_);

  auto beamSpotHandle = getHandle(iEvent, beamSpotToken_);
  GlobalPoint primVtxPos = convertToGP(beamSpotHandle->position());

  // Loop over all super-cluster collections (typically barrel and forward are supplied separately)
  for (const auto& superClustersToken : superClustersTokens_) {
    auto superClustersHandle = getHandle(iEvent, superClustersToken);
    for (auto& superClusRef : *superClustersHandle) {
      //the eta of the supercluster when mustache clustered is slightly biased due to bending in magnetic field
      //the eta of its seed cluster is a better estimate of the orginal position
      GlobalPoint caloPosition(GlobalPoint::Polar(superClusRef->seed()->position().theta(),  //seed theta
                                                  superClusRef->position().phi(),            //supercluster phi
                                                  superClusRef->position().r()));            //supercluster r

      const std::vector<TrajSeedMatcher::SeedWithInfo> matchedSeeds =
          matcher_.compatibleSeeds(*initialSeedsHandle, caloPosition, primVtxPos, superClusRef->energy());

      for (auto& matchedSeed : matchedSeeds) {
        reco::ElectronSeed eleSeed(matchedSeed.seed());
        reco::ElectronSeed::CaloClusterRef caloClusRef(superClusRef);
        eleSeed.setCaloCluster(caloClusRef);
        eleSeed.setNrLayersAlongTraj(matchedSeed.nrValidLayers());
        for (auto& matchInfo : matchedSeed.matches()) {
          eleSeed.addHitInfo(makeSeedPixelVar(matchInfo, *trackerTopoHandle));
        }
        eleSeeds->emplace_back(eleSeed);
      }
    }
  }
  iEvent.put(std::move(eleSeeds));
}

DEFINE_FWK_MODULE(ElectronNHitSeedProducer);
