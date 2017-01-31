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

#include "CommonTools/CandAlgos/interface/ModifyObjectValueBase.h"

#include "RecoEgamma/EgammaTools/interface/GainSwitchTools.h"

#include <iostream>
#include <string>

class GsfElectronGSCrysFixer : public edm::stream::EDProducer<> {
public:
  explicit GsfElectronGSCrysFixer(const edm::ParameterSet& );
  virtual ~GsfElectronGSCrysFixer(){}
  
  void produce(edm::Event&, const edm::EventSetup& ) override;
  void beginLuminosityBlock(edm::LuminosityBlock const&, 
			    edm::EventSetup const&) override;
  

  template<typename T>
  void getToken(edm::EDGetTokenT<T>& token,const edm::ParameterSet& pset,const std::string& label, const std::string& instance = ""){
    auto tag(pset.getParameter<edm::InputTag>(label));
    if (!instance.empty())
      tag = edm::InputTag(tag.label(), instance, tag.process());

    token = consumes<T>(tag);
  }
private:
  typedef edm::View<reco::GsfElectron> GsfElectronView;
  typedef edm::ValueMap<reco::SuperClusterRef> SCRefMap;

  edm::EDGetTokenT<GsfElectronView> oldGsfElesToken_;
  edm::EDGetTokenT<reco::GsfElectronCoreCollection> newCoresToken_;
  edm::EDGetTokenT<SCRefMap> newCoresToOldSCMapToken_;
  edm::EDGetTokenT<EcalRecHitCollection> ebRecHitsToken_;
  
  const CaloTopology* topology_;  
  const CaloGeometry* geometry_;
  std::unique_ptr<ModifyObjectValueBase> gedRegression_;
};


namespace {
  template<typename T> edm::Handle<T> getHandle(const edm::Event& iEvent,const edm::EDGetTokenT<T>& token){
    edm::Handle<T> handle;
    iEvent.getByToken(token,handle);
    return handle;
  }
}



typedef edm::ValueMap<reco::GsfElectronRef> ElectronRefMap;

GsfElectronGSCrysFixer::GsfElectronGSCrysFixer( const edm::ParameterSet & pset )
{
  getToken(oldGsfElesToken_,pset,"oldEles");
  getToken(newCoresToken_,pset,"newCores");
  getToken(newCoresToOldSCMapToken_, pset, "newCores");
  getToken(ebRecHitsToken_,pset,"ebRecHits");

  //ripped wholesale from GEDGsfElectronFinalizer
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

  produces<reco::GsfElectronCollection >();
  // new -> old
  produces<ElectronRefMap>();
}

namespace {
  
  reco::GsfElectronCoreRef getNewCore(reco::GsfElectron const& oldEle,
				      edm::Handle<reco::GsfElectronCoreCollection>& newCores,
				      edm::ValueMap<reco::SuperClusterRef> const& newToOldCoresMap)
  {
    for(size_t coreNr=0;coreNr<newCores->size();coreNr++){
      reco::GsfElectronCoreRef coreRef(newCores,coreNr);
      auto oldRef = newToOldCoresMap[coreRef];
      if( oldRef.isNonnull() && oldRef==oldEle.superCluster()){
	return coreRef;
      }
    }
    return reco::GsfElectronCoreRef(newCores.id());
  }
}

void GsfElectronGSCrysFixer::produce( edm::Event & iEvent, const edm::EventSetup & iSetup )
{
  auto outEles = std::make_unique<reco::GsfElectronCollection>();
 
  if( gedRegression_ ) {
    gedRegression_->setEvent(iEvent);
    gedRegression_->setEventContent(iSetup);
  }

  auto elesHandle = getHandle(iEvent,oldGsfElesToken_);
  auto& ebRecHits = *getHandle(iEvent,ebRecHitsToken_);
  auto newCoresHandle = getHandle(iEvent,newCoresToken_);
  auto newCoresToOldSCMap = *getHandle(iEvent, newCoresToOldSCMapToken_);

  std::vector<reco::GsfElectronRef> oldElectrons;

  unsigned iE(0);
  for (auto& oldEle : *elesHandle) {
    auto&& refToBase(elesHandle->refAt(iE++));
    oldElectrons.emplace_back(refToBase.id(), &oldEle, refToBase.key());

    reco::GsfElectronCoreRef newCoreRef = getNewCore(oldEle,newCoresHandle,newCoresToOldSCMap);

    outEles->emplace_back(oldEle,newCoreRef);
    auto& newEle(outEles->back());

    if (GainSwitchTools::hasEBGainSwitchIn5x5(*oldEle.superCluster(), &ebRecHits, topology_)) {

      reco::GsfElectron::ShowerShape full5x5ShowerShape = GainSwitchTools::redoEcalShowerShape<true>(newEle.full5x5_showerShape(),newEle.superCluster(),&ebRecHits,topology_,geometry_);
      reco::GsfElectron::ShowerShape showerShape = GainSwitchTools::redoEcalShowerShape<false>(newEle.showerShape(),newEle.superCluster(),&ebRecHits,topology_,geometry_);
    
      float eNewSCOverEOldSC = newEle.superCluster()->energy()/oldEle.superCluster()->energy();
      GainSwitchTools::correctHadem(showerShape,eNewSCOverEOldSC,GainSwitchTools::ShowerShapeType::Fractions);
      GainSwitchTools::correctHadem(full5x5ShowerShape,eNewSCOverEOldSC,GainSwitchTools::ShowerShapeType::Full5x5);

      newEle.full5x5_setShowerShape(full5x5ShowerShape);   
      newEle.setShowerShape(showerShape);   


      if( gedRegression_ )
	gedRegression_->modifyObject(newEle);
    }
  }
  
  auto&& newElectronsHandle(iEvent.put(std::move(outEles)));

  std::auto_ptr<ElectronRefMap> pRefMap(new ElectronRefMap);
  ElectronRefMap::Filler refMapFiller(*pRefMap);
  refMapFiller.insert(newElectronsHandle, oldElectrons.begin(), oldElectrons.end());
  refMapFiller.fill();
  iEvent.put(pRefMap);
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



DEFINE_FWK_MODULE(GsfElectronGSCrysFixer);
#endif

