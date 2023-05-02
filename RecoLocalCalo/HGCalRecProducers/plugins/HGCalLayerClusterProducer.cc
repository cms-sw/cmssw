// Authors: Olivie Franklova - olivie.abigail.franklova@cern.ch
// Date: 03/2023
// @file create layer clusters
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

#include "FWCore/Framework/interface/ConsumesCollector.h"

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

  /**
   * @brief Sets algoId accordingly to the detector type
  */
  void setAlgoId();

  /**
   * @brief Counts position for all points in the cluster
   * 
   * @param[in] hitmap hitmap to find correct RecHit
   * @param[in] hitsAndFraction all hits in the cluster
   * @return counted position
  */
  math::XYZPoint calculatePosition(std::unordered_map<uint32_t, const HGCRecHit*>& hitmap,
                                   const std::vector<std::pair<DetId, float>>& hitsAndFractions);

  /**
   * @brief Counts time for all points in the cluster
   * 
   * @param[in] hitmap hitmap to find correct RecHit only for silicon (not for BH-HSci)
   * @param[in] hitsAndFraction all hits in the cluster
   * @return counted time
  */
  std::pair<float, float> calculateTime(std::unordered_map<uint32_t, const HGCRecHit*>& hitmap,
                                        const std::vector<std::pair<DetId, float>>& hitsAndFractions,
                                        size_t sizeCluster);
};

DEFINE_FWK_MODULE(HGCalLayerClusterProducer);

HGCalLayerClusterProducer::HGCalLayerClusterProducer(const edm::ParameterSet& ps)
    : algoId_(reco::CaloCluster::undefined),
      detector_(ps.getParameter<std::string>("detector")),  // one of EE, FH, BH, HFNose
      timeClname_(ps.getParameter<std::string>("timeClname")),
      hitsTime_(ps.getParameter<unsigned int>("nHitsTime")),
      caloGeomToken_(consumesCollector().esConsumes<CaloGeometry, CaloGeometryRecord>()) {
  setAlgoId();  //sets algo id according to detector type
  hits_token_ = consumes<HGCRecHitCollection>(ps.getParameter<edm::InputTag>("recHits"));

  auto pluginPSet = ps.getParameter<edm::ParameterSet>("plugin");
  if (detector_ == "HFNose") {
    algo_ = HGCalLayerClusterAlgoFactory::get()->create("HFNoseCLUE", pluginPSet);
    algo_->setAlgoId(algoId_, true);
  } else {
    algo_ = HGCalLayerClusterAlgoFactory::get()->create(
        pluginPSet.getParameter<std::string>("type"), pluginPSet);
    algo_->setAlgoId(algoId_);
  }
  thresholdW0_ = pluginPSet.getParameter<std::vector<double>>("thresholdW0");
  positionDeltaRho2_ = pluginPSet.getParameter<double>("positionDeltaRho2");

  produces<std::vector<float>>("InitialLayerClustersMask");
  produces<std::vector<reco::BasicCluster>>();
  //time for layer clusters
  produces<edm::ValueMap<std::pair<float, float>>>(timeClname_);
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
  descriptions.add("hgcalLayerClusters", desc);
}

math::XYZPoint HGCalLayerClusterProducer::calculatePosition(
    std::unordered_map<uint32_t, const HGCRecHit*>& hitmap,
    const std::vector<std::pair<DetId, float>>& hitsAndFractions) {
  float total_weight = 0.f;
  float maxEnergyValue = 0.f;
  DetId maxEnergyIndex;
  float x = 0.f;
  float y = 0.f;

  for (auto const& hit : hitsAndFractions) {
    //time is computed wrt  0-25ns + offset and set to -1 if no time
    const HGCRecHit* rechit = hitmap[hit.first];
    total_weight += rechit->energy();
    if (rechit->energy() > maxEnergyValue) {
      maxEnergyValue = rechit->energy();
      maxEnergyIndex = rechit->detid();
    }
  }
  float total_weight_log = 0.f;
  auto thick = rhtools_.getSiThickIndex(maxEnergyIndex);
  const GlobalPoint positionMaxEnergy(rhtools_.getPosition(maxEnergyIndex));
  for (auto const& hit : hitsAndFractions) {
    //time is computed wrt  0-25ns + offset and set to -1 if no time
    const HGCRecHit* rechit = hitmap[hit.first];

    const GlobalPoint position(rhtools_.getPosition(rechit->detid()));

    if (thick != -1) {  //silicon
      //for silicon only just use 1+6 cells = 1.3cm for all thicknesses
      const float d1 = position.x() - positionMaxEnergy.x();
      const float d2 = position.y() - positionMaxEnergy.y();
      if ((d1 * d1 + d2 * d2) > positionDeltaRho2_)
        continue;

      float Wi = std::max(thresholdW0_[thick] + std::log(rechit->energy() / total_weight), 0.);
      x += position.x() * Wi;
      y += position.y() * Wi;
      total_weight_log += Wi;
    } else {  //scintillator
      x += position.x() * rechit->energy();
      y += position.y() * rechit->energy();
    }
  }
  if (thick != -1) {
    total_weight = total_weight_log;
  }
  if (total_weight != 0.) {
    float inv_tot_weight = 1.f / total_weight;
    return math::XYZPoint(x * inv_tot_weight, y * inv_tot_weight, positionMaxEnergy.z());
  } else {
    return math::XYZPoint(0.f, 0.f, 0.f);
  }
}

std::pair<float, float> HGCalLayerClusterProducer::calculateTime(
    std::unordered_map<uint32_t, const HGCRecHit*>& hitmap,
    const std::vector<std::pair<DetId, float>>& hitsAndFractions,
    size_t sizeCluster) {
  std::pair<float, float> timeCl(-99., -1.);

  if (sizeCluster >= hitsTime_) {
    std::vector<float> timeClhits;
    std::vector<float> timeErrorClhits;

    for (auto const& hit : hitsAndFractions) {
      //time is computed wrt  0-25ns + offset and set to -1 if no time
      const HGCRecHit* rechit = hitmap[hit.first];

      float rhTimeE = rechit->timeError();
      //check on timeError to exclude scintillator
      if (rhTimeE < 0.)
        continue;
      timeClhits.push_back(rechit->time());
      timeErrorClhits.push_back(1. / (rhTimeE * rhTimeE));
    }
    hgcalsimclustertime::ComputeClusterTime timeEstimator;
    timeCl = timeEstimator.fixSizeHighestDensity(timeClhits, timeErrorClhits, hitsTime_);
  }
  return timeCl;
}
void HGCalLayerClusterProducer::produce(edm::Event& evt, const edm::EventSetup& es) {
  edm::Handle<HGCRecHitCollection> hits;

  std::unique_ptr<std::vector<reco::BasicCluster>> clusters(new std::vector<reco::BasicCluster>);

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
  *clusters = algo_->getClusters(false);

  std::vector<std::pair<float, float>> times;
  times.reserve(clusters->size());

  for (unsigned i = 0; i < clusters->size(); ++i) {
    const reco::CaloCluster& sCl = (*clusters)[i];
    (*clusters)[i].setPosition(calculatePosition(hitmap, sCl.hitsAndFractions()));
    if (detector_ != "BH") {
      times.push_back(calculateTime(hitmap, sCl.hitsAndFractions(), sCl.size()));
    } else {
      times.push_back(std::pair<float, float>(-99., -1.));
    }
  }

  auto clusterHandle = evt.put(std::move(clusters));

  if (detector_ == "HFNose") {
    std::unique_ptr<std::vector<float>> layerClustersMask(new std::vector<float>);
    layerClustersMask->resize(clusterHandle->size(), 1.0);
    evt.put(std::move(layerClustersMask), "InitialLayerClustersMask");
  }

  auto timeCl = std::make_unique<edm::ValueMap<std::pair<float, float>>>();
  edm::ValueMap<std::pair<float, float>>::Filler filler(*timeCl);
  filler.insert(clusterHandle, times.begin(), times.end());
  filler.fill();
  evt.put(std::move(timeCl), timeClname_);

  algo_->reset();
}

void HGCalLayerClusterProducer::setAlgoId() {
  if (detector_ == "HFNose") {
    algoId_ = reco::CaloCluster::hfnose;
  } else if (detector_ == "EE") {
    algoId_ = reco::CaloCluster::hgcal_em;
  } else {  //for FH or BH
    algoId_ = reco::CaloCluster::hgcal_had;
  }
}
#endif  //__RecoLocalCalo_HGCRecProducers_HGCalLayerClusterProducer_H__
