#include "RecoEgamma/EgammaPhotonProducers/interface/GEDPhotonGSCrysFixer.h"

namespace {
  template<typename T> edm::Handle<T> getHandle(const edm::Event& iEvent,const edm::EDGetTokenT<T>& token){
    edm::Handle<T> handle;
    iEvent.getByToken(token,handle);
    return handle;
  }
}


GEDPhotonGSCrysFixer::GEDPhotonGSCrysFixer( const edm::ParameterSet & pset )
{
  getToken(newCoresToken_,pset,"newCores");
  getToken(oldGedPhosToken_,pset,"oldPhos");
  getToken(ebRecHitsToken_,pset,"ebRecHits");
  getToken(newCoresToOldCoresMapToken_,pset,"newCoresToOldCoresMap");

  //ripped wholesale from GEDPhotonFinalizer
  if( pset.existsAs<edm::ParameterSet>("regressionConfig") ) {
    const edm::ParameterSet& iconf = pset.getParameterSet("regressionConfig");
    const std::string& mname = iconf.getParameter<std::string>("modifierName");
    ModifyObjectValueBase* plugin = 
    ModifyObjectValueFactory::get()->create(mname,iconf);
    gedRegression_.reset(plugin);
    edm::ConsumesCollector sumes = consumesCollector();
    gedRegression_->setConsumes(sumes);
  } else {
    gedRegression_.reset(nullptr);
  }

  produces<reco::PhotonCollection >();
}

namespace {
  
  reco::PhotonCoreRef getNewCore(const reco::PhotonRef& oldPho,
				 edm::Handle<reco::PhotonCoreCollection>& newCores,
				 edm::ValueMap<reco::SuperClusterRef> newToOldCoresMap)
  {
    for(size_t coreNr=0;coreNr<newCores->size();coreNr++){
      reco::PhotonCoreRef coreRef(newCores,coreNr);
      auto oldRef = newToOldCoresMap[coreRef];
      if( oldRef.isNonnull() && oldRef==oldPho->superCluster()){
	return coreRef;
      }
    }
    return reco::PhotonCoreRef(newCores.id());
  }
}

void GEDPhotonGSCrysFixer::produce( edm::Event & iEvent, const edm::EventSetup & iSetup )
{
  auto outPhos = std::make_unique<reco::PhotonCollection>();
 
  if( gedRegression_ ) {
    gedRegression_->setEvent(iEvent);
    gedRegression_->setEventContent(iSetup);
  }
 

  auto phosHandle = getHandle(iEvent,oldGedPhosToken_);
  auto& ebRecHits = *getHandle(iEvent,ebRecHitsToken_);
  auto& newCoresToOldCoresMap = *getHandle(iEvent,newCoresToOldCoresMapToken_);
  auto newCoresHandle = getHandle(iEvent,newCoresToken_);
  
  for(size_t phoNr=0;phoNr<phosHandle->size();phoNr++){
    reco::PhotonRef phoRef(phosHandle,phoNr);
  
    reco::PhotonCoreRef newCoreRef = getNewCore(phoRef,newCoresHandle,newCoresToOldCoresMap);
    
    if(newCoreRef.isNonnull()){ //okay we have to remake the photon

      reco::Photon newPho(*phoRef);
      newPho.setPhotonCore(newCoreRef);

      reco::Photon::ShowerShape full5x5ShowerShape = GainSwitchTools::redoEcalShowerShape<true>(newPho.full5x5_showerShapeVariables(),newPho.superCluster(),phoRef->superCluster(),&ebRecHits,topology_,geometry_);
      reco::Photon::ShowerShape ShowerShape = GainSwitchTools::redoEcalShowerShape<false>(newPho.showerShapeVariables(),newPho.superCluster(),phoRef->superCluster(),&ebRecHits,topology_,geometry_);
      newPho.full5x5_setShowerShapeVariables(full5x5ShowerShape);   
      newPho.setShowerShapeVariables(ShowerShape);   

      if( gedRegression_ ) {
	gedRegression_->modifyObject(newPho);
      }

      outPhos->push_back(newPho);
    }else{
      outPhos->push_back(*phoRef);
    }
  }
  
  iEvent.put(std::move(outPhos));
}

void GEDPhotonGSCrysFixer::beginLuminosityBlock(edm::LuminosityBlock const& lb, 
						 edm::EventSetup const& es) {
  edm::ESHandle<CaloGeometry> caloGeom ;
  edm::ESHandle<CaloTopology> caloTopo ;
  es.get<CaloGeometryRecord>().get(caloGeom);
  es.get<CaloTopologyRecord>().get(caloTopo);
  geometry_ = caloGeom.product();
  topology_ = caloTopo.product();
}



