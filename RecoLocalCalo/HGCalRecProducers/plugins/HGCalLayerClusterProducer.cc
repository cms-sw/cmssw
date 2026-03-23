// Authors: Olivie Franklova - olivie.abigail.franklova@cern.ch
// Date: 03/2023
// @file create layer clusters

#define DEBUG_CLUSTERS_ALPAKA 0

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/PluginDescription.h"

#include "HeterogeneousCore/AlpakaInterface/interface/host.h"

#include "RecoParticleFlow/PFClusterProducer/interface/RecHitTopologicalCleanerBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/SeedFinderBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/InitialClusteringStepBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFClusterBuilderBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFCPositionCalculatorBase.h"
#include "RecoParticleFlow/PFClusterProducer/interface/PFClusterEnergyCorrectorBase.h"
#include "RecoLocalCalo/HGCalRecProducers/interface/ComputeClusterTime.h"

#include "RecoLocalCalo/HGCalRecProducers/interface/HGCalLayerClusterAlgoFactory.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/HGCalDepthPreClusterer.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"

#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/TICL/interface/CaloClusterHostCollection.h"
#include "DataFormats/TICL/interface/AssociationMap.h"

#if DEBUG_CLUSTERS_ALPAKA
#include "RecoLocalCalo/HGCalRecProducers/interface/DumpClustersDetails.h"
#endif

class HGCalLayerClusterProducer : public edm::stream::EDProducer<> {
public:
  /**
   * @brief Constructor with parameter settings - which can be changed in hgcalLayerCluster_cff.py.
   * Constructor will set all variables by input param ps.
   * algoID variables will be set accordingly to the detector type.
   *
   * @param[in] ps parametr set to set variables
  */
  HGCalLayerClusterProducer(const edm::ParameterSet&);
  ~HGCalLayerClusterProducer() override {}
  /**
   * @brief Method fill description which will be used in pyhton file.
   *
   * @param[out] description to be fill
  */
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  /**
   * @brief Method run the algoritm to get clusters.
   *
   * @param[in, out] evt from get info and put result
   * @param[in] es to get event setup info
  */
  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  edm::EDGetTokenT<HGCRecHitCollection> hits_token_;

  reco::CaloCluster::AlgoId algoId_;

  std::unique_ptr<HGCalClusteringAlgoBase> algo_;
  std::string detector_;

  std::string timeClname_;
  unsigned int hitsTime_;

  // for calculate position
  std::vector<double> thresholdW0_;
  double positionDeltaRho2_;
  hgcal::RecHitTools rhtools_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> caloGeomToken_;
  const bool calculatePositionInAlgo_;
#if DEBUG_CLUSTERS_ALPAKA
  std::string moduleLabel_;
#endif

  /**
   * @brief Sets algoId accordingly to the detector type
  */
  void setAlgoId();

  /**
   * @brief Counts time for all points in the cluster
   *
   * @param[in] hitmap hitmap to find correct RecHit only for silicon (not for BH-HSci)
   * @param[in] hitsAndFraction all hits in the cluster
   * @return counted time
  */
  void calculateTime(std::unordered_map<uint32_t, const HGCRecHit*>& hitmap,
                     reco::CaloClusterHostCollection::View layerClusters,
                     ticl::HitsAndFractionsHost::ConstView hitsAndFractions);
};

HGCalLayerClusterProducer::HGCalLayerClusterProducer(const edm::ParameterSet& ps)
    : algoId_(reco::CaloCluster::undefined),
      detector_(ps.getParameter<std::string>("detector")),  // one of EE, FH, BH, HFNose
      timeClname_(ps.getParameter<std::string>("timeClname")),
      hitsTime_(ps.getParameter<unsigned int>("nHitsTime")),
      caloGeomToken_(consumesCollector().esConsumes<CaloGeometry, CaloGeometryRecord>()),
      calculatePositionInAlgo_(ps.getParameter<bool>("calculatePositionInAlgo")) {
#if DEBUG_CLUSTERS_ALPAKA
  moduleLabel_ = ps.getParameter<std::string>("@module_label");
#endif
  setAlgoId();  //sets algo id according to detector type
  hits_token_ = consumes<HGCRecHitCollection>(ps.getParameter<edm::InputTag>("recHits"));

  auto pluginPSet = ps.getParameter<edm::ParameterSet>("plugin");
  if (detector_ == "HFNose") {
    algo_ = HGCalLayerClusterAlgoFactory::get()->create("HFNoseCLUE", pluginPSet);
    algo_->setAlgoId(algoId_, true);
  } else {
    algo_ = HGCalLayerClusterAlgoFactory::get()->create(pluginPSet.getParameter<std::string>("type"), pluginPSet);
    algo_->setAlgoId(algoId_);
  }
  thresholdW0_ = pluginPSet.getParameter<std::vector<double>>("thresholdW0");
  positionDeltaRho2_ = pluginPSet.getParameter<double>("positionDeltaRho2");

  produces<std::vector<float>>("InitialLayerClustersMask");
  produces<reco::CaloClusterHostCollection>();
  // produces<std::vector<reco::BasicCluster>>();
  //time for layer clusters
  // produces<edm::ValueMap<std::pair<float, float>>>(timeClname_);
}

void HGCalLayerClusterProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // hgcalLayerClusters
  edm::ParameterSetDescription desc;
  edm::ParameterSetDescription pluginDesc;
  pluginDesc.addNode(edm::PluginDescription<HGCalLayerClusterAlgoFactory>("type", "SiCLUE", true));

  desc.add<edm::ParameterSetDescription>("plugin", pluginDesc);
  desc.add<std::string>("detector", "EE")->setComment("options EE, FH, BH,  HFNose; other value defaults to EE");
  desc.add<edm::InputTag>("recHits", edm::InputTag("HGCalRecHit", "HGCEERecHits"));
  desc.add<std::string>("timeClname", "timeLayerCluster");
  desc.add<unsigned int>("nHitsTime", 3);
  desc.add<bool>("calculatePositionInAlgo", true);
  descriptions.add("hgcalLayerClusters", desc);
}

void HGCalLayerClusterProducer::calculateTime(std::unordered_map<uint32_t, const HGCRecHit*>& hitmap,
                                              reco::CaloClusterHostCollection::View layerClusters,
                                              ticl::HitsAndFractionsHost::ConstView hitsAndFractions) {
  for (auto cluster = 0; cluster < hitsAndFractions.keys(); ++cluster) {
    std::pair<float, float> timeCl(-99., -1.);

    if (hitsAndFractions.count(cluster) >= static_cast<int>(hitsTime_)) {
      std::vector<float> timeClhits;
      std::vector<float> timeErrorClhits;

      for (auto const& hitAndFraction : hitsAndFractions[cluster]) {
        //time is computed wrt  0-25ns + offset and set to -1 if no time
        const HGCRecHit* rechit = hitmap[hitAndFraction.hit];

        float rhTimeE = rechit->timeError();
        //check on timeError to exclude scintillator
        if (rhTimeE < 0.f)
          continue;
        timeClhits.push_back(rechit->time());
        timeErrorClhits.push_back(1.f / (rhTimeE * rhTimeE));
      }
      hgcalsimclustertime::ComputeClusterTime timeEstimator;
      timeCl = timeEstimator.fixSizeHighestDensity(timeClhits, timeErrorClhits, hitsTime_);
    }
    layerClusters.timing()[cluster].time() = timeCl.first;
    layerClusters.timing()[cluster].timeError() = timeCl.second;
  }
}
void HGCalLayerClusterProducer::produce(edm::Event& evt, const edm::EventSetup& es) {
  edm::Handle<HGCRecHitCollection> hits;

  edm::ESHandle<CaloGeometry> geom = es.getHandle(caloGeomToken_);
  rhtools_.setGeometry(*geom);
  algo_->getEventSetup(es, rhtools_);

  //make a map detid-rechit
  // NB for the moment just host EE and FH hits
  // timing in digi for BH not implemented for now
  std::unordered_map<uint32_t, const HGCRecHit*> hitmap;

  evt.getByToken(hits_token_, hits);
  algo_->populate(*hits);
  for (auto const& it : *hits) {
    hitmap[it.detid().rawId()] = &(it);
  }

  algo_->makeClusters();

  auto clusters_and_associations = algo_->getClusters(false);
  auto clusters = std::move(clusters_and_associations.layer_clusters);
  auto hits_and_fractions = std::move(clusters_and_associations.hits_and_fractions);

  std::vector<std::pair<float, float>> times;
  times.reserve(clusters->view().position().metadata().size());

  calculateTime(hitmap, clusters->view(), hits_and_fractions->view());
  // for (unsigned i = 0; i < legacy_clusters->size(); ++i) {
  //   reco::CaloCluster& sCl = (*legacy_clusters)[i];
  //   if (detector_ != "BH") {
  //     times.push_back(calculateTime(hitmap, sCl.hitsAndFractions(), sCl.size()));
  //   } else {
  //     times.push_back(std::pair<float, float>(-99.f, -1.f));
  //   }
  // }

#if DEBUG_CLUSTERS_ALPAKA
  hgcalUtils::DumpClusters dumper;
  auto runNumber = evt.eventAuxiliary().run();
  auto lumiNumber = evt.eventAuxiliary().luminosityBlock();
  auto evtNumber = evt.eventAuxiliary().id().event();

  dumper.dumpInfos(*legacy_clusters, moduleLabel_, runNumber, lumiNumber, evtNumber, true);
#endif

  if (detector_ == "HFNose") {
    std::unique_ptr<std::vector<float>> layerClustersMask(new std::vector<float>);
    layerClustersMask->resize(clusters->view().position().metadata().size(), 1.0);
    evt.put(std::move(layerClustersMask), "InitialLayerClustersMask");
  }

  // auto timeCl = std::make_unique<edm::ValueMap<std::pair<float, float>>>();
  // edm::ValueMap<std::pair<float, float>>::Filler filler(*timeCl);
  // filler.insert(*legacy_clusters, times.begin(), times.end());
  // filler.fill();
  // evt.put(std::move(timeCl), timeClname_);

  evt.put(std::move(clusters));
  evt.put(std::move(hits_and_fractions));

  algo_->reset();
}

void HGCalLayerClusterProducer::setAlgoId() {
  if (detector_ == "EE") {
    algoId_ = reco::CaloCluster::hgcal_em;
  } else if (detector_ == "FH") {
    algoId_ = reco::CaloCluster::hgcal_had;
  } else if (detector_ == "BH") {
    algoId_ = reco::CaloCluster::hgcal_scintillator;
  } else if (detector_ == "HFNose") {
    algoId_ = reco::CaloCluster::hfnose;
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HGCalLayerClusterProducer);
