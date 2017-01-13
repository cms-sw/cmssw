#ifndef RecoEgamma_EgammaElectronProducers_GsfElectronGSCrysFixer_h
#define RecoEgamma_EgammaElectronProducers_GsfElectronGSCrysFixer_h

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h" 
#include "FWCore/Framework/interface/ESHandle.h"


#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/Common/interface/Handle.h" 
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronCoreFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronCore.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/CaloTopologyRecord.h"

#include "RecoCaloTools/Navigation/interface/CaloNavigator.h"

#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"
#include "FWCore/Utilities/interface/isFinite.h"


#include <iostream>
#include <string>

class GsfElectronGSCrysFixer : public edm::stream::EDProducer<> {
public:
  explicit GsfElectronGSCrysFixer(const edm::ParameterSet& );
  virtual ~GsfElectronGSCrysFixer(){}
  
  void produce(edm::Event&, const edm::EventSetup& ) override;
  void beginLuminosityBlock(edm::LuminosityBlock const&, 
			    edm::EventSetup const&) override;
  

  reco::GsfElectron::ShowerShape 
  calShowerShape(const reco::SuperClusterRef& superClus, const EcalRecHitCollection* recHits);
  

  template<typename T>
  void getToken(edm::EDGetTokenT<T>& token,const edm::ParameterSet& pset,const std::string& label){
    token=consumes<T>(pset.getParameter<edm::InputTag>(label));
  }
private:
  edm::EDGetTokenT<reco::GsfElectronCollection> oldGsfElesToken_;
  edm::EDGetTokenT<reco::GsfElectronCoreCollection> newCoresToken_;
  edm::EDGetTokenT<EcalRecHitCollection> ebRecHitsToken_;
  edm::EDGetTokenT<edm::ValueMap<reco::SuperClusterRef> > newCoresToOldCoresMapToken_;
  
  const CaloTopology* topology_;  
  const CaloGeometry* geometry_;
  
};


namespace {
  template<typename T> edm::Handle<T> getHandle(const edm::Event& iEvent,const edm::EDGetTokenT<T>& token){
    edm::Handle<T> handle;
    iEvent.getByToken(token,handle);
    return handle;
  }
}





GsfElectronGSCrysFixer::GsfElectronGSCrysFixer( const edm::ParameterSet & pset )
{
  getToken(newCoresToken_,pset,"newCores");
  getToken(oldGsfElesToken_,pset,"oldEles");
  getToken(ebRecHitsToken_,pset,"ebRecHits");
  getToken(newCoresToOldCoresMapToken_,pset,"newCoresToOldCoresMap");
  produces<reco::GsfElectronCollection >();
}
namespace {
  
  reco::GsfElectronCoreRef getNewCore(const reco::GsfElectronRef& oldEle,
				      edm::Handle<reco::GsfElectronCoreCollection>& newCores,
				      edm::ValueMap<reco::SuperClusterRef> newToOldCoresMap)
  {
    for(size_t coreNr=0;coreNr<newCores->size();coreNr++){
      reco::GsfElectronCoreRef coreRef(newCores,coreNr);
      auto oldRef = newToOldCoresMap[coreRef];
      if( oldRef.isNonnull() && oldRef==oldEle->superCluster()){
	return coreRef;
      }
    }
    return reco::GsfElectronCoreRef(newCores.id());
  }
}

void GsfElectronGSCrysFixer::produce( edm::Event & iEvent, const edm::EventSetup & iSetup )
{
  auto outEles = std::make_unique<reco::GsfElectronCollection>();
  

  auto elesHandle = getHandle(iEvent,oldGsfElesToken_);
  auto& ebRecHits = *getHandle(iEvent,ebRecHitsToken_);
  auto& newCoresToOldCoresMap = *getHandle(iEvent,newCoresToOldCoresMapToken_);
  auto newCoresHandle = getHandle(iEvent,newCoresToken_);
  
  for(size_t eleNr=0;eleNr<elesHandle->size();eleNr++){
    reco::GsfElectronRef eleRef(elesHandle,eleNr);
  
    reco::GsfElectronCoreRef newCoreRef = getNewCore(eleRef,newCoresHandle,newCoresToOldCoresMap);
    
    if(newCoreRef.isNonnull()){ //okay we have to remake the electron
      std::cout <<"made a new electron "<<iEvent.id().run()<<" "<<iEvent.id().event()<<std::endl;
      reco::GsfElectron newEle(*eleRef,newCoreRef);
      reco::GsfElectron::ShowerShape full5x5 = calShowerShape(newEle.superCluster(),&ebRecHits);
      newEle.full5x5_setShowerShape(full5x5);   
      
      outEles->push_back(newEle);
    }else{
      outEles->push_back(*eleRef);
    }
  }
  
  iEvent.put(std::move(outEles));
}

void GsfElectronGSCrysFixer::beginLuminosityBlock(edm::LuminosityBlock const& lb, 
						 edm::EventSetup const& es) {
  edm::ESHandle<CaloGeometry> caloGeom ;
  edm::ESHandle<CaloTopology> caloTopo ;
  es.get<CaloGeometryRecord>().get(caloGeom);
  es.get<CaloTopologyRecord>().get(caloTopo);
  geometry_ = caloGeom.product();
  topology_ = caloTopo.product();
}


reco::GsfElectron::ShowerShape 
GsfElectronGSCrysFixer::calShowerShape(const reco::SuperClusterRef& superClus, const EcalRecHitCollection* recHits)
{
  reco::GsfElectron::ShowerShape showerShape;
  
  const reco::CaloCluster & seedClus = *(superClus->seed());
  
  std::vector<float> covariances = noZS::EcalClusterTools::covariances(seedClus,recHits,topology_,geometry_);
  std::vector<float> localCovariances = noZS::EcalClusterTools::localCovariances(seedClus,recHits,topology_);
  showerShape.sigmaEtaEta = sqrt(covariances[0]);
  showerShape.sigmaIetaIeta = sqrt(localCovariances[0]);
  if (!edm::isNotFinite(localCovariances[2])) showerShape.sigmaIphiIphi = sqrt(localCovariances[2]);
  showerShape.e1x5 = noZS::EcalClusterTools::e1x5(seedClus,recHits,topology_);
  showerShape.e2x5Max = noZS::EcalClusterTools::e2x5Max(seedClus,recHits,topology_);
  showerShape.e5x5 = noZS::EcalClusterTools::e5x5(seedClus,recHits,topology_);
  showerShape.r9 = noZS::EcalClusterTools::e3x3(seedClus,recHits,topology_)/superClus->rawEnergy();
  return showerShape;
}



DEFINE_FWK_MODULE(GsfElectronGSCrysFixer);
#endif

