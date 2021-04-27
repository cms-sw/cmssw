#ifndef RecoLocalTracker_SiStripClusters2ApproxClustersv1_h
#define RecoLocalTracker_SiStripClusters2ApproxClustersv1_h

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
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

private:

  edm::InputTag inputClusters;
  edm::EDGetTokenT< edmNew::DetSetVector<SiStripCluster> > clusterToken;  
};

DEFINE_FWK_MODULE(SiStripClusters2ApproxClustersv1);
#endif

