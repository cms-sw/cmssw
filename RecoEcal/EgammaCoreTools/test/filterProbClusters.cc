// -*- C++ -*-



#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// to access recHits and BasicClusters
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"

// to use the cluster tools
#include "FWCore/Framework/interface/ESHandle.h"

#include "RecoEcal/EgammaCoreTools/interface/EcalClusterSeverityLevelAlgo.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgoRcd.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"

#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"

class ProbClustersFilter : public edm::EDFilter {
public:
  explicit ProbClustersFilter(const edm::ParameterSet&);
  ~ProbClustersFilter();
private:
  virtual bool filter(edm::Event&, const edm::EventSetup&);
private:
  int maxDistance_;
  float maxGoodFraction_;
  edm::ParameterSet conf_;
  edm::InputTag barrelClusterCollection_;
  edm::InputTag endcapClusterCollection_;
  edm::InputTag reducedBarrelRecHitCollection_;
  edm::InputTag reducedEndcapRecHitCollection_;
 
};

ProbClustersFilter::ProbClustersFilter(const edm::ParameterSet& iConfig) {
  maxDistance_ = iConfig.getParameter<int>("maxDistance");
  maxGoodFraction_ = iConfig.getParameter<double>("maxGoodFraction");
  barrelClusterCollection_ = iConfig.getParameter<edm::InputTag>("barrelClusterCollection");
  endcapClusterCollection_ = iConfig.getParameter<edm::InputTag>("endcapClusterCollection");
  reducedBarrelRecHitCollection_ = iConfig.getParameter<edm::InputTag>("reducedBarrelRecHitCollection");
  reducedEndcapRecHitCollection_ = iConfig.getParameter<edm::InputTag>("reducedEndcapRecHitCollection");
  conf_ = iConfig;
 
}


ProbClustersFilter::~ProbClustersFilter() {}

bool ProbClustersFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) 
{
  int problematicClusters=0;

  edm::Handle< reco::SuperClusterCollection > pEBClusters;
  iEvent.getByLabel( barrelClusterCollection_, pEBClusters );
  const reco::SuperClusterCollection *ebClusters = pEBClusters.product();
  
  //edm::Handle< reco::SuperClusterCollection > pEEClusters;
  //iEvent.getByLabel( endcapClusterCollection_, pEEClusters );
  //const reco::SuperClusterCollection *eeClusters = pEEClusters.product();
  
  edm::Handle< EcalRecHitCollection > pEBRecHits;
  iEvent.getByLabel( reducedBarrelRecHitCollection_, pEBRecHits );
  const EcalRecHitCollection *ebRecHits = pEBRecHits.product();
  
  //edm::Handle< EcalRecHitCollection > pEERecHits;
  //iEvent.getByLabel( reducedEndcapRecHitCollection_, pEERecHits );
  //const EcalRecHitCollection *eeRecHits = pEERecHits.product();
  
  //edm::ESHandle<CaloGeometry> pGeometry;
  //iSetup.get<CaloGeometryRecord>().get(pGeometry);
  //const CaloGeometry *geometry = pGeometry.product();
  
  edm::ESHandle<CaloTopology> pTopology;
  iSetup.get<CaloTopologyRecord>().get(pTopology);
  const CaloTopology *topology = pTopology.product();
  
  
  edm::ESHandle<EcalSeverityLevelAlgo> sevlvh;  
  iSetup.get<EcalSeverityLevelAlgoRcd>().get(sevlvh);  
  const EcalSeverityLevelAlgo * sevLv= sevlvh.product();

  //        std::cout << "========== BARREL ==========" << std::endl;
  for (reco::SuperClusterCollection::const_iterator it = ebClusters->begin(); it != ebClusters->end(); ++it ) 
    {
      float goodFraction=EcalClusterSeverityLevelAlgo::goodFraction( *it, *ebRecHits,*sevLv);
      std::pair<int,int> distance=EcalClusterSeverityLevelAlgo::etaphiDistanceClosestProblematic( *it, *ebRecHits,topology,*sevLv);
      if ( distance.first == -1 && distance.second==-1)
	{
	  distance.first=999;
	  distance.second=999;
	};
      if ( goodFraction >= maxGoodFraction_ && sqrt(distance.first*distance.first +distance.second*distance.second) >= maxDistance_ )
	continue;
      ++problematicClusters;
      //       std::cout << "seed" << EBDetId(EcalClusterTools::getMaximum(*it, ebRecHits).first) << std::endl;
      //       std::cout << "goodFraction" << EcalClusterSeverityLevelAlgo::goodFraction( *it, *ebRecHits, *theEcalChStatus) << std::endl;
      //       std::cout << "closestProblematicDetId" << EBDetId(EcalClusterSeverityLevelAlgo::closestProblematic( *it, *ebRecHits, *theEcalChStatus,topology)) << std::endl;
      //       std::cout << "(deta,dphi)" << "(" << EcalClusterSeverityLevelAlgo::etaphiDistanceClosestProblematic( *it, *ebRecHits, *theEcalChStatus,topology).first << "," <<EcalClusterSeverityLevelAlgo::etaphiDistanceClosestProblematic( *it, *ebRecHits, *theEcalChStatus,topology).second << ")" << std::endl;
    }

  return problematicClusters;
}

DEFINE_FWK_MODULE(ProbClustersFilter);

