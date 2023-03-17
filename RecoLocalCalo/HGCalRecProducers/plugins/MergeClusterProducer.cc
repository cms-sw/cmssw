#ifndef __RecoLocalCalo_HGCRecProducers_MergeClusterProducer_H__
#define __RecoLocalCalo_HGCRecProducers_MergeClusterProducer_H__
// Authors: Olivie Franklova - olivie.abigail.franklova@cern.ch
// Date: 03/2023
// @file merge layer clusters which were produce by HGCalLayerClusterProducer

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/OrphanHandle.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/PluginDescription.h"

// #include "RecoParticleFlow/PFClusterProducer/interface/RecHitTopologicalCleanerBase.h"
// #include "RecoParticleFlow/PFClusterProducer/interface/SeedFinderBase.h"
// #include "RecoParticleFlow/PFClusterProducer/interface/InitialClusteringStepBase.h"
// #include "RecoParticleFlow/PFClusterProducer/interface/PFClusterBuilderBase.h"
// #include "RecoParticleFlow/PFClusterProducer/interface/PFCPositionCalculatorBase.h"
// #include "RecoParticleFlow/PFClusterProducer/interface/PFClusterEnergyCorrectorBase.h"
#include "RecoLocalCalo/HGCalRecProducers/interface/ComputeClusterTime.h"

#include "RecoLocalCalo/HGCalRecAlgos/interface/HGCalDepthPreClusterer.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"

#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/Common/interface/ValueMap.h"


class MergeClusterProducer : public edm::stream::EDProducer<> {
public:
  /**
   * @brief Constructor with parameter settings - which can be changed in  ...todo.
   * Constructor will set all variables by input param ps. 
   * 
   * @param[in] ps parametr set to set variables
  */
  MergeClusterProducer(const edm::ParameterSet&);
  ~MergeClusterProducer() override {}
  /**
   * @brief Method fill description which will be used in pyhton file.
   * 
   * @param[out] description to be fill
  */
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  /**
   * @brief Method will merge the producers and put them back to event
   * 
   * @param[in, out] evt from get info and put result
   * @param[in] es to get event setup info
  */
  void produce(edm::Event&, const edm::EventSetup&) override;

  private:
  typedef std::map<DetId, float> Density;
  //todo make them const
  edm::EDGetTokenT<std::vector<reco::CaloCluster>> EEclusters_token_;
  edm::EDGetTokenT<std::vector<reco::CaloCluster>> HSiclusters_token_;
  edm::EDGetTokenT<std::vector<reco::CaloCluster>> HSciclusters_token_;

  edm::EDGetTokenT<Density> EEdensity_token_;
  edm::EDGetTokenT<Density> HSidensity_token_;
  edm::EDGetTokenT<Density> HScidensity_token_;

  std::string timeClname_;
  const edm::EDGetTokenT<edm::ValueMap<std::pair<float, float>>> clustersTimeEE_token_;
  const edm::EDGetTokenT<edm::ValueMap<std::pair<float, float>>> clustersTimeHSi_token_;
  const edm::EDGetTokenT<edm::ValueMap<std::pair<float, float>>> clustersTimeHSci_token_;
  // OrphanHandle<std::vector<reco::CaloCluster>> clusterHandle; todo

  /**
   * @brief method merge three vectors of reco::CaloCluster to one
   * 
   * @param[out] merge the vector into which others vectors will be merge
   * @param[in] EE vector for Electromagnetic silicon
   * @param[in] HSi vector for Hardon silicon
   * @param[in] ESci vector for hadron scintillator
  */
  void mergeTogether(std::vector<reco::CaloCluster> &merge, const std::vector<reco::CaloCluster> &EE, 
                    const std::vector<reco::CaloCluster> &HSi, const std::vector<reco::CaloCluster> &HSci);
  /**
   * @brief method merge three Densities (std::map<DetId, float>) to one
   * 
   * @param[out] merge the Density into which others Density will be merge
   * @param[in] EE Density for Electromagnetic silicon
   * @param[in] HSi Density for Hardon silicon
   * @param[in] ESci Density for hadron scintillator
  */
  void mergeTogether(Density  &merge, const Density &EE, const Density &HSi, const Density &HSci);

  void addTo(std::vector<std::pair<float, float>> &to, const edm::ValueMap<std::pair<float, float>> &vm){
    size_t size = vm.size();
    for (size_t i = 0; i < size; ++i){
      to.push_back(vm.get(i));
    }
  }

  void mergeTime(edm::Event& evt, size_t size, std::vector<std::pair<float, float>>& times){
    edm::Handle<edm::ValueMap<std::pair<float, float>>> EE, HSi, HSci;
    // get values from all three part of detectors
    evt.getByToken(clustersTimeEE_token_, EE);
    evt.getByToken(clustersTimeHSi_token_, HSi);
    evt.getByToken(clustersTimeHSci_token_, HSci);
    
    times.reserve(size);
    addTo(times, *EE);
    addTo(times, *HSi);
    addTo(times, *HSci);
  }
  /**
   * @brief get info form event and then call merge
   * 
   * it is used for merge density and clusters and time
   * 
   * @param[in] evt Event
   * @param[in] EE_token token for Electromagnetic silicon
   * @param[in] HSi_token token for Hardon silicon
   * @param[in] ESci_token token for hadron scintillator
   * @return merged result
  */
  template <typename T>
  void createMerge(edm::Event& evt, const edm::EDGetTokenT<T> &EE_token,
   const edm::EDGetTokenT<T> &HSi_token, const edm::EDGetTokenT<T> &HSci_token, T & merge){
    edm::Handle<T> EE, HSi, HSci;
    // get values from all three part of detectors
    evt.getByToken(EE_token, EE);
    evt.getByToken(HSi_token, HSi);
    evt.getByToken(HSci_token, HSci);
    mergeTogether(merge, *EE, *HSi, *HSci );
  }

};

DEFINE_FWK_MODULE(MergeClusterProducer); 

MergeClusterProducer::MergeClusterProducer(const edm::ParameterSet& ps)
    : timeClname_(ps.getParameter<std::string>("timeClname")),
      clustersTimeEE_token_(consumes<edm::ValueMap<std::pair<float, float>>>(ps.getParameter<edm::InputTag>("time_layerclustersEE"))),
      clustersTimeHSi_token_(consumes<edm::ValueMap<std::pair<float, float>>>(ps.getParameter<edm::InputTag>("time_layerclustersHSi"))),
      clustersTimeHSci_token_(consumes<edm::ValueMap<std::pair<float, float>>>(ps.getParameter<edm::InputTag>("time_layerclustersHSci")))
    {
    EEclusters_token_ = consumes<std::vector<reco::CaloCluster>>(ps.getParameter<edm::InputTag>("layerClustersEE"));
    HSiclusters_token_ = consumes<std::vector<reco::CaloCluster>>(ps.getParameter<edm::InputTag>("layerClustersHSi"));
    HSciclusters_token_ = consumes<std::vector<reco::CaloCluster>>(ps.getParameter<edm::InputTag>("layerClustersHSci"));

    EEdensity_token_ = consumes<Density>(ps.getParameter<edm::InputTag>("densityEE"));
    HSidensity_token_ = consumes<Density>(ps.getParameter<edm::InputTag>("densityHSi"));
    HScidensity_token_ = consumes<Density>(ps.getParameter<edm::InputTag>("densityHSci"));

    produces<std::vector<float>>("InitialLayerClustersMask");
    produces<std::vector<reco::BasicCluster>>();
    produces<std::vector<reco::BasicCluster>>("sharing");
    //density
    produces<Density>();
    //time for layer clusters
    produces<edm::ValueMap<std::pair<float, float>>>(timeClname_);
}

void MergeClusterProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // hgcalMergeLayerClusters
  edm::ParameterSetDescription desc;
  //layer clusters
  desc.add<edm::InputTag>("layerClustersEE", edm::InputTag( "hgcalLayerClustersEE"));
  desc.add<edm::InputTag>("layerClustersHSi", edm::InputTag( "hgcalLayerClustersHSi"));
  desc.add<edm::InputTag>("layerClustersHSci", edm::InputTag( "hgcalLayerClustersHSci"));

  //density
  //todo could be like that??
  desc.add<edm::InputTag>("densityEE", edm::InputTag("hgcalLayerClustersEE"));
  desc.add<edm::InputTag>("densityHSi", edm::InputTag("hgcalLayerClustersHSi"));
  desc.add<edm::InputTag>("densityHSci", edm::InputTag("hgcalLayerClustersHSci"));
  //time
  desc.add<edm::InputTag>("time_layerclustersEE", edm::InputTag( "hgcalLayerClustersEE", "timeLayerCluster"));
  desc.add<edm::InputTag>("time_layerclustersHSi", edm::InputTag( "hgcalLayerClustersHSi", "timeLayerCluster"));
  desc.add<edm::InputTag>("time_layerclustersHSci", edm::InputTag( "hgcalLayerClustersHSci", "timeLayerCluster"));

  desc.add<std::string>("timeClname", "timeLayerCluster");
  descriptions.add("hgcalMergeLayerClusters", desc);
}

void MergeClusterProducer::produce(edm::Event& evt, const edm::EventSetup& es){

  //merge clusters
  std::unique_ptr<std::vector<reco::BasicCluster>> clusters(new std::vector<reco::BasicCluster>);
  createMerge(evt, EEclusters_token_, HSiclusters_token_, HSciclusters_token_, *clusters );
  //put new clusters to event
  auto clusterHandle  = evt.put(std::move(clusters));

  //merge densities
  auto density = std::make_unique<Density>();
  createMerge(evt, EEdensity_token_, HSidensity_token_, HScidensity_token_, *density);
  //put merged density to event
  evt.put(std::move(density));

  //create layer cluster mask
  std::unique_ptr<std::vector<float>> layerClustersMask(new std::vector<float>);
  layerClustersMask->resize(clusterHandle->size(), 1.0);
  //put it into event
  evt.put(std::move(layerClustersMask), "InitialLayerClustersMask");

  //time
  std::vector<std::pair<float, float>> times;
  mergeTime(evt, clusterHandle->size(), times);

  auto timeCl = std::make_unique<edm::ValueMap<std::pair<float, float>>>();
  edm::ValueMap<std::pair<float, float>>::Filler filler(*timeCl);
  filler.insert(clusterHandle, times.begin(), times.end());
  filler.fill();
  evt.put(std::move(timeCl), timeClname_);
  


}
void MergeClusterProducer::mergeTogether(std::vector<reco::CaloCluster> &merge, const std::vector<reco::CaloCluster> &EE, 
                    const std::vector<reco::CaloCluster> &HSi, const std::vector<reco::CaloCluster> &HSci){
    auto clusterSize = EE.size() + HSi.size() + HSci.size();
    merge.reserve(clusterSize);
  
    merge.insert(merge.end(), EE.begin(), EE.end());
    merge.insert(merge.end(), HSi.begin(), HSi.end());
    merge.insert(merge.end(), HSci.begin(), HSci.end());


}

 void MergeClusterProducer::mergeTogether(Density  &merge, const Density &EE, const Density &HSi, const Density &HSci){
  // merge is in this order HSci(BH), HSi(FH), EE, because in original one is it inserted in reverse order so if tere will be any same values in EE they will be rewrited by FH or BH
    merge.insert(HSci.begin(), HSci.end());
    merge.insert(HSi.begin(), HSi.end());
    merge.insert(EE.begin(), EE.end());
}

#endif
