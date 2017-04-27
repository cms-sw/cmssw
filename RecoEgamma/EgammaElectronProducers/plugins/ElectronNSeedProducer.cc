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
#include "DataFormats/EgammaReco/interface/ElectronNHitSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronNHitSeedFwd.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "RecoEgamma/EgammaElectronAlgos/interface/PixelNHitMatcher.h"

class ElectronNSeedProducer : public edm::stream::EDProducer<> {
public:
  
  
  explicit ElectronNSeedProducer( const edm::ParameterSet & ) ;
  virtual ~ElectronNSeedProducer()=default;  
  
  virtual void produce( edm::Event &, const edm::EventSetup & ) override final;
  
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:

  PixelNHitMatcher matcher_;
  
  std::vector<edm::EDGetTokenT<std::vector<reco::SuperClusterRef>> > superClustersTokens_;
  edm::EDGetTokenT<TrajectorySeedCollection> initialSeedsToken_ ;
  edm::EDGetTokenT<std::vector<reco::Vertex> > verticesToken_;
  edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_ ;
  edm::EDGetTokenT<MeasurementTrackerEvent> measTkEvtToken_;
  
};

namespace {
  template<typename T> 
  edm::Handle<T> getHandle(const edm::Event& event,const edm::EDGetTokenT<T>& token)
  {
    edm::Handle<T> handle;
    event.getByToken(token,handle);
    return handle;
  }

  template<typename T>
  GlobalPoint convertToGP(const T& orgPoint){
    return GlobalPoint(orgPoint.x(),orgPoint.y(),orgPoint.z());
  }

  int getLayerOrDiskNr(DetId detId,const TrackerTopology& trackerTopo)
  {
    if(detId.subdetId()==PixelSubdetector::PixelBarrel){
      return trackerTopo.pxbLayer(detId);
    }else if(detId.subdetId()==PixelSubdetector::PixelEndcap){
      return trackerTopo.pxfDisk(detId);
    }else return -1;
  }
  
  
  reco::ElectronNHitSeed::PMVars 
  makeSeedPixelVar(const PixelNHitMatcher::MatchInfo& matchInfo,
		   const TrackerTopology& trackerTopo)
  {
    
    int layerOrDisk = getLayerOrDiskNr(matchInfo.detId,trackerTopo);
    reco::ElectronNHitSeed::PMVars pmVars;
    pmVars.setDet(matchInfo.detId,layerOrDisk);
    pmVars.setDPhi(matchInfo.dPhiPos,matchInfo.dPhiNeg);
    pmVars.setDRZ(matchInfo.dRZPos,matchInfo.dRZNeg);
    
    return pmVars;
  }  

}

ElectronNSeedProducer::ElectronNSeedProducer( const edm::ParameterSet& pset):
  matcher_(pset.getParameter<edm::ParameterSet>("matcherConfig")),
  initialSeedsToken_(consumes<TrajectorySeedCollection>(pset.getParameter<edm::InputTag>("initialSeeds"))),
  verticesToken_(consumes<std::vector<reco::Vertex> >(pset.getParameter<edm::InputTag>("vertices"))),
  beamSpotToken_(consumes<reco::BeamSpot>(pset.getParameter<edm::InputTag>("beamSpot"))),
  measTkEvtToken_(consumes<MeasurementTrackerEvent>(pset.getParameter<edm::InputTag>("measTkEvt")))
{
  const auto superClusTags = pset.getParameter<std::vector<edm::InputTag> >("superClusters");
  for(const auto& scTag : superClusTags){
    superClustersTokens_.emplace_back(consumes<std::vector<reco::SuperClusterRef>>(scTag));
  }
  produces<reco::ElectronNHitSeedCollection>() ;
}

void ElectronNSeedProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions)
{
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("initialSeeds",edm::InputTag());
  desc.add<edm::InputTag>("vertices",edm::InputTag());
  desc.add<edm::InputTag>("beamSpot",edm::InputTag()); 
  desc.add<edm::InputTag>("measTkEvt",edm::InputTag());
  desc.add<std::vector<edm::InputTag> >("superClusters");
  desc.add<edm::ParameterSetDescription>("matcherConfig",PixelNHitMatcher::makePSetDescription());
  
  descriptions.add("electronNSeedProducer",desc);
}

void ElectronNSeedProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{ 
  edm::ESHandle<TrackerTopology> trackerTopoHandle;
  iSetup.get<TrackerTopologyRcd>().get(trackerTopoHandle);
  
  matcher_.doEventSetup(iSetup);
  matcher_.setMeasTkEvtHandle(getHandle(iEvent,measTkEvtToken_));

  auto eleSeeds = std::make_unique<reco::ElectronNHitSeedCollection>();
  
  auto initialSeedsHandle = getHandle(iEvent,initialSeedsToken_);

  auto beamSpotHandle = getHandle(iEvent,beamSpotToken_);
  GlobalPoint primVtxPos = convertToGP(beamSpotHandle->position());

  for(const auto& superClustersToken : superClustersTokens_){
    auto superClustersHandle = getHandle(iEvent,superClustersToken);
    for(auto& superClusRef : *superClustersHandle){
      const std::vector<PixelNHitMatcher::SeedWithInfo> matchedSeeds = 
	matcher_.compatibleSeeds(*initialSeedsHandle,convertToGP(superClusRef->position()),
				 primVtxPos,superClusRef->energy());
      
      for(auto& matchedSeed : matchedSeeds){
	reco::ElectronNHitSeed eleSeed(matchedSeed.seed()); 
	reco::ElectronNHitSeed::CaloClusterRef caloClusRef(superClusRef);
	eleSeed.setCaloCluster(caloClusRef);
	eleSeed.setNrLayersAlongTraj(matchedSeed.nrValidLayers());
	for(auto& matchInfo : matchedSeed.matches()){
	  eleSeed.addHitInfo(makeSeedPixelVar(matchInfo,*trackerTopoHandle));
	}
	eleSeeds->emplace_back(eleSeed);
      }
    }
    
  }
  iEvent.put(std::move(eleSeeds));
}
  

  
DEFINE_FWK_MODULE(ElectronNSeedProducer);
