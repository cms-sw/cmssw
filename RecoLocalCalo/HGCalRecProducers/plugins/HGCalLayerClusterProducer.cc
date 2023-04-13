#ifndef __RecoLocalCalo_HGCRecProducers_HGCalLayerClusterProducer_H__
#define __RecoLocalCalo_HGCRecProducers_HGCalLayerClusterProducer_H__

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/PluginDescription.h"

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
  edm::EDGetTokenT<HGCRecHitCollection> hits_token;

  reco::CaloCluster::AlgoId algoId;

  std::unique_ptr<HGCalClusteringAlgoBase> algo;
  std::string detector;

  std::string timeClname;
  unsigned int nHitsTime;

  /**
   * @brief Sets algoId accordingly to the detector type
  */
  void setAlgoId();
};

DEFINE_FWK_MODULE(HGCalLayerClusterProducer);

HGCalLayerClusterProducer::HGCalLayerClusterProducer(const edm::ParameterSet& ps)
    : algoId(reco::CaloCluster::undefined),
      detector(ps.getParameter<std::string>("detector")),  // one of EE, FH, BH, HFNose
      timeClname(ps.getParameter<std::string>("timeClname")),
      nHitsTime(ps.getParameter<unsigned int>("nHitsTime")) {

  setAlgoId(); //sets algo id according to detector type
  hits_token = consumes<HGCRecHitCollection>(ps.getParameter<edm::InputTag>("recHits"));

  auto pluginPSet = ps.getParameter<edm::ParameterSet>("plugin");
  if (detector == "HFNose") {
    algo = HGCalLayerClusterAlgoFactory::get()->create("HFNoseCLUE", pluginPSet, consumesCollector());
    algo->setAlgoId(algoId, true);
  } else {
    algo = HGCalLayerClusterAlgoFactory::get()->create(
        pluginPSet.getParameter<std::string>("type"), pluginPSet, consumesCollector());
    algo->setAlgoId(algoId);
  }

  produces<std::vector<float>>("InitialLayerClustersMask");
  produces<std::vector<reco::BasicCluster>>();
  //time for layer clusters
  produces<edm::ValueMap<std::pair<float, float>>>(timeClname);
}

void HGCalLayerClusterProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // hgcalLayerClusters
  edm::ParameterSetDescription desc;
  edm::ParameterSetDescription pluginDesc;
  pluginDesc.addNode(edm::PluginDescription<HGCalLayerClusterAlgoFactory>("type", "SiCLUE", true));

  desc.add<edm::ParameterSetDescription>("plugin", pluginDesc);
  desc.add<std::string>("detector", "EE")
      ->setComment("options EE, FH, BH,  HFNose; other value defaults to EE");
  desc.add<edm::InputTag>("recHits", edm::InputTag("HGCalRecHit", "HGCEERecHits"));
  desc.add<std::string>("timeClname", "timeLayerCluster");
  desc.add<unsigned int>("nHitsTime", 3);
  descriptions.add("hgcalLayerClusters", desc);
}

void HGCalLayerClusterProducer::produce(edm::Event& evt, const edm::EventSetup& es) {
  edm::Handle<HGCRecHitCollection> hits;

  std::unique_ptr<std::vector<reco::BasicCluster>> clusters(new std::vector<reco::BasicCluster>);

  algo->getEventSetup(es);

  //make a map detid-rechit
  // NB for the moment just host EE and FH hits
  // timing in digi for BH not implemented for now
  std::unordered_map<uint32_t, const HGCRecHit*> hitmap;

  evt.getByToken(hits_token, hits);
  algo->populate(*hits);
  if (detector != "BH") {
    for (auto const& it : *hits)
      hitmap[it.detid().rawId()] = &(it);
  }
  algo->makeClusters();
  *clusters = algo->getClusters(false);
  auto clusterHandle = evt.put(std::move(clusters));

  edm::PtrVector<reco::BasicCluster> clusterPtrs;

  std::vector<std::pair<float, float>> times;
  times.reserve(clusterHandle->size());

  for (unsigned i = 0; i < clusterHandle->size(); ++i) {
    edm::Ptr<reco::BasicCluster> ptr(clusterHandle, i);
    clusterPtrs.push_back(ptr);

    std::pair<float, float> timeCl(-99., -1.);

    const reco::CaloCluster& sCl = (*clusterHandle)[i];
    if (sCl.size() >= nHitsTime) {
      std::vector<float> timeClhits;
      std::vector<float> timeErrorClhits;

      for (auto const& hit : sCl.hitsAndFractions()) {
        auto finder = hitmap.find(hit.first);
        if (finder == hitmap.end())
          continue;

        //time is computed wrt  0-25ns + offset and set to -1 if no time
        const HGCRecHit* rechit = finder->second;
        float rhTimeE = rechit->timeError();
        //check on timeError to exclude scintillator
        if (rhTimeE < 0.)
          continue;
        timeClhits.push_back(rechit->time());
        timeErrorClhits.push_back(1. / (rhTimeE * rhTimeE));
      }
      hgcalsimclustertime::ComputeClusterTime timeEstimator;
      timeCl = timeEstimator.fixSizeHighestDensity(timeClhits, timeErrorClhits, nHitsTime);
    }
    times.push_back(timeCl);
  }
  if (detector == "HFNose"){
    std::unique_ptr<std::vector<float>> layerClustersMask(new std::vector<float>);
    layerClustersMask->resize(clusterHandle->size(), 1.0);
    evt.put(std::move(layerClustersMask), "InitialLayerClustersMask");
  }

  auto timeCl = std::make_unique<edm::ValueMap<std::pair<float, float>>>();
  edm::ValueMap<std::pair<float, float>>::Filler filler(*timeCl);
  filler.insert(clusterHandle, times.begin(), times.end());
  filler.fill();
  evt.put(std::move(timeCl), timeClname);

  algo->reset();
}

// todo or we can make a map but I dont think it is necessary
void HGCalLayerClusterProducer::setAlgoId(){
    if (detector == "HFNose") {
      algoId = reco::CaloCluster::hfnose;
    }  else if (detector == "EE") {
      algoId = reco::CaloCluster::hgcal_em;
    } else { //for FH or BH
     algoId = reco::CaloCluster::hgcal_had;
    }
}
#endif  //__RecoLocalCalo_HGCRecProducers_HGCalLayerClusterProducer_H__
