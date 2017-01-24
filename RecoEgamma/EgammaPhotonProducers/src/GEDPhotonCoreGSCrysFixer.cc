#include "RecoEgamma/EgammaPhotonProducers/interface/GEDPhotonCoreGSCrysFixer.h"

namespace {
  template<typename T> edm::Handle<T> getHandle(const edm::Event& iEvent,const edm::EDGetTokenT<T>& token){
    edm::Handle<T> handle;
    iEvent.getByToken(token,handle);
    return handle;
  }
}


GEDPhotonCoreGSCrysFixer::GEDPhotonCoreGSCrysFixer( const edm::ParameterSet & pset )
{
  getToken(orgCoresToken_,pset,"orgCores");
  getToken(ebRecHitsToken_,pset,"ebRecHits");
  getToken(oldRefinedSCToNewMapToken_,pset,"oldRefinedSCToNewMap");
  getToken(oldSCToNewMapToken_,pset,"oldSCToNewMap");
  
  produces<reco::PhotonCoreCollection >();
  produces<edm::ValueMap<reco::SuperClusterRef> >("parentCores");
}


void GEDPhotonCoreGSCrysFixer::produce( edm::Event & iEvent, const edm::EventSetup & iSetup )
{
  auto outCores = std::make_unique<reco::PhotonCoreCollection>();
  
  auto phoCoresHandle = getHandle(iEvent,orgCoresToken_);
  auto& ebRecHits = *getHandle(iEvent,ebRecHitsToken_);
  auto& oldRefinedSCToNewMap = *getHandle(iEvent,oldRefinedSCToNewMapToken_);
  auto& oldSCToNewMap = *getHandle(iEvent,oldSCToNewMapToken_);
  
  std::vector<reco::SuperClusterRef> parentCores;

  for(size_t coreNr=0;coreNr<phoCoresHandle->size();coreNr++){
    reco::PhotonCoreRef coreRef(phoCoresHandle,coreNr);
    const reco::PhotonCore& core = *coreRef;
   
    //passing in the old refined supercluster
    if(GainSwitchTools::hasEBGainSwitchIn5x5(*core.superCluster(),&ebRecHits,topology_)){
      reco::PhotonCore newCore(core);
      //these references may be null, lets see what happens!     
      //turns out the orginal pho may have null references, odd
      if(core.superCluster().isNonnull()) newCore.setSuperCluster( oldRefinedSCToNewMap[core.superCluster()] );
      if(core.parentSuperCluster().isNonnull()) newCore.setParentSuperCluster( oldSCToNewMap[core.parentSuperCluster()] );
        
      outCores->push_back(newCore);
      parentCores.push_back(coreRef->superCluster());
    }
  }
  
  auto outCoresHandle = iEvent.put(std::move(outCores));
  
  std::unique_ptr<edm::ValueMap<reco::SuperClusterRef> > parentCoresValMap(new edm::ValueMap<reco::SuperClusterRef>());
  typename edm::ValueMap<reco::SuperClusterRef>::Filler filler(*parentCoresValMap);
  filler.insert(outCoresHandle, parentCores.begin(), parentCores.end());
  filler.fill();
  iEvent.put(std::move(parentCoresValMap),"parentCores");
}

void GEDPhotonCoreGSCrysFixer::beginLuminosityBlock(edm::LuminosityBlock const& lb, 
						      edm::EventSetup const& es) {
  edm::ESHandle<CaloTopology> caloTopo ;
  es.get<CaloTopologyRecord>().get(caloTopo);
  topology_ = caloTopo.product();
}


