#ifndef RecoEgamma_EgammaElectronProducers_GsfElectronCoreGSCrysFixer_h
#define RecoEgamma_EgammaElectronProducers_GsfElectronCoreGSCrysFixer_h

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
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/Records/interface/CaloTopologyRecord.h"

#include "RecoCaloTools/Navigation/interface/CaloNavigator.h"

#include "RecoEgamma/EgammaTools/interface/GainSwitchTools.h"

#include <iostream>
#include <string>

class GsfElectronCoreGSCrysFixer : public edm::stream::EDProducer<> {
public:
  explicit GsfElectronCoreGSCrysFixer(const edm::ParameterSet& );
  virtual ~GsfElectronCoreGSCrysFixer(){}
  
  void produce(edm::Event&, const edm::EventSetup& ) override;
  void beginLuminosityBlock(edm::LuminosityBlock const&, 
			    edm::EventSetup const&) override;
  

  template<typename T>
  void getToken(edm::EDGetTokenT<T>& token,const edm::ParameterSet& pset,const std::string& label){
    token=consumes<T>(pset.getParameter<edm::InputTag>(label));
  }
private:
  edm::EDGetTokenT<reco::GsfElectronCoreCollection> orgCoresToken_;
  edm::EDGetTokenT<EcalRecHitCollection> ebRecHitsToken_;
  edm::EDGetTokenT<edm::ValueMap<reco::SuperClusterRef> > oldRefinedSCToNewMapToken_;
  edm::EDGetTokenT<edm::ValueMap<reco::SuperClusterRef> > oldSCToNewMapToken_;
  const CaloTopology* topology_;
  
};


namespace {
  template<typename T> edm::Handle<T> getHandle(const edm::Event& iEvent,const edm::EDGetTokenT<T>& token){
    edm::Handle<T> handle;
    iEvent.getByToken(token,handle);
    return handle;
  }
}


GsfElectronCoreGSCrysFixer::GsfElectronCoreGSCrysFixer( const edm::ParameterSet & pset )
{
  getToken(orgCoresToken_,pset,"orgCores");
  getToken(ebRecHitsToken_,pset,"ebRecHits");
  getToken(oldRefinedSCToNewMapToken_,pset,"oldRefinedSCToNewMap");
  getToken(oldSCToNewMapToken_,pset,"oldSCToNewMap");
  
  produces<reco::GsfElectronCoreCollection >();
  produces<edm::ValueMap<reco::SuperClusterRef> >("parentCores");
}


void GsfElectronCoreGSCrysFixer::produce( edm::Event & iEvent, const edm::EventSetup & iSetup )
{
  auto outCores = std::make_unique<reco::GsfElectronCoreCollection>();
  
  auto eleCoresHandle = getHandle(iEvent,orgCoresToken_);
  auto& ebRecHits = *getHandle(iEvent,ebRecHitsToken_);
  auto& oldRefinedSCToNewMap = *getHandle(iEvent,oldRefinedSCToNewMapToken_);
  auto& oldSCToNewMap = *getHandle(iEvent,oldSCToNewMapToken_);
  
  std::vector<reco::SuperClusterRef> parentCores;

  for(size_t coreNr=0;coreNr<eleCoresHandle->size();coreNr++){
    reco::GsfElectronCoreRef coreRef(eleCoresHandle,coreNr);
    const reco::GsfElectronCore& core = *coreRef;
   
    //passing in the old refined supercluster
    if(GainSwitchTools::hasEBGainSwitchIn5x5(*core.superCluster(),&ebRecHits,topology_)){
      reco::GsfElectronCore newCore(core);
      //these references may be null, lets see what happens!     
      //turns out the orginal ele may have null references, odd
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

void GsfElectronCoreGSCrysFixer::beginLuminosityBlock(edm::LuminosityBlock const& lb, 
						      edm::EventSetup const& es) {
  edm::ESHandle<CaloTopology> caloTopo ;
  es.get<CaloTopologyRecord>().get(caloTopo);
  topology_ = caloTopo.product();
}


DEFINE_FWK_MODULE(GsfElectronCoreGSCrysFixer);
#endif
