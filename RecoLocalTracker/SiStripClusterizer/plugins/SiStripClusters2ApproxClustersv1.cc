

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/SiStripCluster/interface/SiStripApproximateClusterv1.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/DetSetVector.h"


#include <vector>
#include <memory>

class SiStripClusters2ApproxClustersv1: public edm::stream::EDProducer<>  {

public:

  explicit SiStripClusters2ApproxClustersv1(const edm::ParameterSet& conf);
  void produce(edm::Event&, const edm::EventSetup&) override;
  
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:

  edm::InputTag inputClusters;
  edm::EDGetTokenT< edmNew::DetSetVector<SiStripCluster> > clusterToken;  
};



SiStripClusters2ApproxClustersv1::SiStripClusters2ApproxClustersv1(const edm::ParameterSet& conf){
  //inputClusters = conf.getParameter< edm::InputTag >("inputClusters");
  auto cc = setWhatProduced(this, conf.getParameter< edm::InputTag >("inputClusters"));

  //clusterToken = consumes< edmNew::DetSetVector< SiStripCluster > >(inputClusters);
  clusterToken = cc.consumes< edmNew::DetSetVector< SiStripCluster > >(inputClusters);
  produces< edmNew::DetSetVector< SiStripApproximateClusterv1 > >(); 

}

void SiStripClusters2ApproxClustersv1::produce(edm::Event& e, edm::EventSetup const&){
  auto result = std::make_unique<edmNew::DetSetVector< SiStripApproximateClusterv1 > >();
  edm::Handle<edmNew::DetSetVector< SiStripCluster >> clusterCollection = e.getHandle(clusterToken);


  for ( const auto& detClusters : *clusterCollection ) {
    std::vector< SiStripApproximateClusterv1 > tempVec;    
    edmNew::DetSetVector<SiStripApproximateClusterv1>::FastFiller ff{*result, detClusters.id()};

    for ( const auto& cluster : detClusters ) ff.push_back(SiStripApproximateClusterv1(cluster));
    
  }

  e.put(std::move(result));
}

void
SiStripClusters2ApproxClustersv1::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("inputClusters", edm::InputTag("siStripClusters"));
  descriptions.add("SiStripClusters2ApproxClustersv1", desc);  
}


DEFINE_FWK_MODULE(SiStripClusters2ApproxClustersv1);