#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/allowedValues.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/HGCalReco/interface/HGCalSoAClustersHostCollection.h"
#include "DataFormats/HGCalReco/interface/HGCalSoARecHitsExtraHostCollection.h"
#include "DataFormats/HGCalReco/interface/HGCalSoARecHitsHostCollection.h"

#include "RecoLocalCalo/HGCalRecProducers/interface/ComputeClusterTime.h"

#define DEBUG_CLUSTERS_ALPAKA 0

#if DEBUG_CLUSTERS_ALPAKA
#include "RecoLocalCalo/HGCalRecProducers/interface/DumpClustersDetails.h"
#endif

class HGCalLayerClustersFromSoAProducer : public edm::stream::EDProducer<> {
public:
  HGCalLayerClustersFromSoAProducer(edm::ParameterSet const& config)
      : getTokenSoAClusters_(consumes(config.getParameter<edm::InputTag>("src"))),
        getTokenSoACells_(consumes(config.getParameter<edm::InputTag>("hgcalRecHitsSoA"))),
        getTokenSoARecHitsExtra_(consumes(config.getParameter<edm::InputTag>("hgcalRecHitsLayerClustersSoA"))),
        detector_(config.getParameter<std::string>("detector")),
        hitsTime_(config.getParameter<unsigned int>("nHitsTime")),
        timeClname_(config.getParameter<std::string>("timeClname")) {
#if DEBUG_CLUSTERS_ALPAKA
    moduleLabel_ = config.getParameter<std::string>("@module_label");
#endif
    if (detector_ == "HFNose") {
      algoId_ = reco::CaloCluster::hfnose;
    } else if (detector_ == "EE") {
      algoId_ = reco::CaloCluster::hgcal_em;
    } else {  //for FH or BH
      algoId_ = reco::CaloCluster::hgcal_had;
    }

    produces<std::vector<float>>("InitialLayerClustersMask");
    produces<std::vector<reco::BasicCluster>>();
    produces<edm::ValueMap<std::pair<float, float>>>(timeClname_);
  }

  ~HGCalLayerClustersFromSoAProducer() override = default;

  void produce(edm::Event& iEvent, edm::EventSetup const& iSetup) override {
    auto const& deviceData = iEvent.get(getTokenSoAClusters_);

    auto const& deviceSoARecHitsExtra = iEvent.get(getTokenSoARecHitsExtra_);
    auto const soaRecHitsExtra_v = deviceSoARecHitsExtra.view();

    auto const& deviceSoACells = iEvent.get(getTokenSoACells_);
    auto const soaCells_v = deviceSoACells.view();

    auto const deviceView = deviceData.view();

    std::unique_ptr<std::vector<reco::BasicCluster>> clusters(new std::vector<reco::BasicCluster>);
    clusters->reserve(deviceData->metadata().size());

    // Create a vector of <clusters> locations, where each location holds a
    // vector of <nCells> floats. These vectors are used to compute the time for
    // each cluster.
    std::vector<std::vector<float>> times(deviceData->metadata().size());
    std::vector<std::vector<float>> timeErrors(deviceData->metadata().size());

    for (int i = 0; i < deviceData->metadata().size(); ++i) {
      std::vector<std::pair<DetId, float>> thisCluster;
      thisCluster.reserve(deviceView.cells(i));
      clusters->emplace_back(deviceView.energy(i),
                             math::XYZPoint(deviceView.x(i), deviceView.y(i), deviceView.z(i)),
                             reco::CaloID::DET_HGCAL_ENDCAP,
                             std::move(thisCluster),
                             algoId_);
      clusters->back().setSeed(deviceView.seed(i));
      times[i].reserve(deviceView.cells(i));
      timeErrors[i].reserve(deviceView.cells(i));
    }

    // Populate hits and fractions required to compute the cluster's time.
    // This procedure is complex and involves two SoAs: the original RecHits
    // SoA and the clustering algorithm's output SoA. Both SoAs have the same
    // cardinality, and crucially, the output SoA includes the cluster index.
    for (int32_t i = 0; i < soaRecHitsExtra_v.metadata().size(); ++i) {
      if (soaRecHitsExtra_v[i].clusterIndex() == -1) {
        continue;
      }
      assert(soaRecHitsExtra_v[i].clusterIndex() < (int)clusters->size());
      (*clusters)[soaRecHitsExtra_v[i].clusterIndex()].addHitAndFraction(soaCells_v[i].detid(), 1.f);
      if (soaCells_v[i].timeError() < 0.f) {
        continue;
      }
      times[soaRecHitsExtra_v[i].clusterIndex()].push_back(soaCells_v[i].time());
      timeErrors[soaRecHitsExtra_v[i].clusterIndex()].push_back(
          1.f / (soaCells_v[i].timeError() * soaCells_v[i].timeError()));
    }

    // Finally, compute and assign the time to each cluster.
    std::vector<std::pair<float, float>> cluster_times;
    cluster_times.reserve(clusters->size());
    hgcalsimclustertime::ComputeClusterTime timeEstimator;
    for (unsigned i = 0; i < clusters->size(); ++i) {
      if (detector_ != "BH") {
        cluster_times.push_back(timeEstimator.fixSizeHighestDensity(times[i], timeErrors[i], hitsTime_));
      } else {
        cluster_times.push_back(std::pair<float, float>(-99.f, -1.f));
      }
    }

#if DEBUG_CLUSTERS_ALPAKA
    auto runNumber = iEvent.eventAuxiliary().run();
    auto lumiNumber = iEvent.eventAuxiliary().luminosityBlock();
    auto evtNumber = iEvent.eventAuxiliary().id().event();

    hgcalUtils::DumpCellsSoA dumperCellsSoA;
    dumperCellsSoA.dumpInfos(deviceSoACells, moduleLabel_, runNumber, lumiNumber, evtNumber);

    hgcalUtils::DumpClusters dumper;
    dumper.dumpInfos(*clusters, moduleLabel_, runNumber, lumiNumber, evtNumber, true);

    hgcalUtils::DumpClustersSoA dumperSoA;
    dumperSoA.dumpInfos(deviceSoARecHitsExtra, moduleLabel_, runNumber, lumiNumber, evtNumber);
#endif

    auto clusterHandle = iEvent.put(std::move(clusters));

    auto timeCl = std::make_unique<edm::ValueMap<std::pair<float, float>>>();
    edm::ValueMap<std::pair<float, float>>::Filler filler(*timeCl);
    filler.insert(clusterHandle, cluster_times.begin(), cluster_times.end());
    filler.fill();
    iEvent.put(std::move(timeCl), timeClname_);

    // The layerClusterMask for the HGCAL detector is created at a later
    // stage, when the layer clusters from the different components of HGCAL
    // are merged together into a unique collection. For the case of HFNose,
    // since there is no further merging step needed, we create the
    // layerClustersMask directly here.
    if (detector_ == "HFNose") {
      std::unique_ptr<std::vector<float>> layerClustersMask(new std::vector<float>);
      layerClustersMask->resize(clusterHandle->size(), 1.0);
      iEvent.put(std::move(layerClustersMask), "InitialLayerClustersMask");
    }
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("src", edm::InputTag("hltHgcalSoALayerClustersProducer"));
    desc.add<edm::InputTag>("hgcalRecHitsLayerClustersSoA", edm::InputTag("hltHgcalSoARecHitsLayerClustersProducer"));
    desc.add<edm::InputTag>("hgcalRecHitsSoA", edm::InputTag("hltHgcalSoARecHitsProducer"));
    desc.add<unsigned int>("nHitsTime", 3);
    desc.add<std::string>("timeClname", "timeLayerCluster");
    desc.ifValue(edm::ParameterDescription<std::string>(
                     "detector", "EE", true, edm::Comment("the HGCAL component used to create clusters.")),
                 edm::allowedValues<std::string>("EE", "FH"));
    descriptions.addWithDefaultLabel(desc);
  }

private:
  edm::EDGetTokenT<HGCalSoAClustersHostCollection> const getTokenSoAClusters_;
  edm::EDGetTokenT<HGCalSoARecHitsHostCollection> const getTokenSoACells_;
  edm::EDGetTokenT<HGCalSoARecHitsExtraHostCollection> const getTokenSoARecHitsExtra_;
  std::string detector_;
  unsigned int hitsTime_;
  std::string timeClname_;
  reco::CaloCluster::AlgoId algoId_;
#if DEBUG_CLUSTERS_ALPAKA
  std::string moduleLabel_;
#endif
};
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HGCalLayerClustersFromSoAProducer);
