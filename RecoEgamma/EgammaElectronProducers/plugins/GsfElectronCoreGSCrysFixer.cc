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

int nrCrysWithFlagsIn5x5(const DetId& id,const std::vector<int>& flags,const EcalRecHitCollection* recHits,const CaloTopology *topology)
{
  int nrFound=0;
  CaloNavigator<DetId> cursor = CaloNavigator<DetId>( id, topology->getSubdetectorTopology( id ) );
  
  for ( int eastNr = -2; eastNr <= 2; ++eastNr ) { //east is eta in barrel
    for ( int northNr = -2; northNr <= 2; ++northNr ) { //north is phi in barrel
      cursor.home();
      cursor.offsetBy( eastNr, northNr);
      DetId id = *cursor;
      auto recHitIt = recHits->find(id);
      if(recHitIt!=recHits->end() && 
	 recHitIt->checkFlags(flags)){
	nrFound++;
      }
		
    }
  }
  return nrFound;
}




GsfElectronCoreGSCrysFixer::GsfElectronCoreGSCrysFixer( const edm::ParameterSet & pset )
{
  getToken(orgCoresToken_,pset,"orgCores");
  getToken(ebRecHitsToken_,pset,"ebRecHits");
  getToken(oldRefinedSCToNewMapToken_,pset,"oldRefinedSCToNewMap");
  getToken(oldSCToNewMapToken_,pset,"oldSCToNewMap");
  
  produces<reco::GsfElectronCoreCollection >();
}


void GsfElectronCoreGSCrysFixer::produce( edm::Event & iEvent, const edm::EventSetup & iSetup )
{
  auto outCores = std::make_unique<reco::GsfElectronCoreCollection>();
  
  auto& eleCores = *getHandle(iEvent,orgCoresToken_);
  auto& ebRecHits = *getHandle(iEvent,ebRecHitsToken_);
  auto& oldRefinedSCToNewMap = *getHandle(iEvent,oldRefinedSCToNewMapToken_);
  auto& oldSCToNewMap = *getHandle(iEvent,oldSCToNewMapToken_);
  
  for( const auto& core : eleCores ) {
    int nrEBGSCrys=0;
    DetId seedId = core.superCluster()->seed()->seed();
    if(seedId.subdetId()==EcalBarrel){
      nrEBGSCrys = nrCrysWithFlagsIn5x5(seedId,
					{EcalRecHit::kHasSwitchToGain6,EcalRecHit::kHasSwitchToGain1}, 
					&ebRecHits,topology_);
    }
    
    if(nrEBGSCrys>0){ //okay we have to remake the electron core
      reco::GsfElectronCore newCore(core);
      
      //these references may be null, lets see what happens!
      newCore.setSuperCluster( oldRefinedSCToNewMap[core.superCluster()] );
      newCore.setParentSuperCluster( oldSCToNewMap[core.parentSuperCluster()] );
      
      outCores->push_back(core);
    }else{
      outCores->push_back(core);
    }
  }
  
  iEvent.put(std::move(outCores));
}

void GsfElectronCoreGSCrysFixer::beginLuminosityBlock(edm::LuminosityBlock const& lb, 
						      edm::EventSetup const& es) {
  edm::ESHandle<CaloTopology> caloTopo ;
  es.get<CaloTopologyRecord>().get(caloTopo);
  topology_ = caloTopo.product();
}


DEFINE_FWK_MODULE(GsfElectronCoreGSCrysFixer);
#endif
