#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"

#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/PatCandidates/interface/Photon.h"
#include "DataFormats/PatCandidates/interface/Electron.h"

#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

#include "RecoEgamma/EgammaTools/interface/GainSwitchTools.h"

#include <memory>
#include <vector>

namespace {
  template<typename T>
  edm::Handle<T>
  getHandle(const edm::Event& iEvent, const edm::EDGetTokenT<T>& token)
  {
    edm::Handle<T> handle;
    iEvent.getByToken(token, handle);
    return handle;
  }
}

typedef edm::ValueMap<int> intMap;

template<typename C>
class EGGainSwitchFlagProducer : public edm::stream::EDProducer<> {

public:
  
  EGGainSwitchFlagProducer(const edm::ParameterSet&);
  ~EGGainSwitchFlagProducer();
  
  void produce(edm::Event&, const edm::EventSetup&) override;
  void beginLuminosityBlock(edm::LuminosityBlock const&,
                            edm::EventSetup const&) override;

  template<typename T>
  void getToken(edm::EDGetTokenT<T>& token,const edm::ParameterSet& pset, const std::string& label, const std::string& instance = "") {
    auto tag(pset.getParameter<edm::InputTag>(label));
    if (!instance.empty())
      tag = edm::InputTag(tag.label(), instance, tag.process());
    
    token = consumes<T>(tag);
  }
  
private:

  const CaloTopology* topology_;  
  edm::EDGetTokenT<EcalRecHitCollection>   ebRecHitsToken_;
  edm::EDGetTokenT<C> objectsToken_;
  std::string hasGainSwitchFlag_;

};

template<typename C>
EGGainSwitchFlagProducer<C>::EGGainSwitchFlagProducer(const edm::ParameterSet& iConfig) :
  hasGainSwitchFlag_("hasGainSwitchFlag")

{

  getToken(ebRecHitsToken_,iConfig,"ebRecHits");
  getToken(objectsToken_,iConfig,"src");

  produces<intMap>(hasGainSwitchFlag_);  

}

template<typename C>
EGGainSwitchFlagProducer<C>::~EGGainSwitchFlagProducer() {
}

template<typename C>
void EGGainSwitchFlagProducer<C>::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {


  auto ebRecHits = *getHandle(iEvent, ebRecHitsToken_);
  auto objects   = getHandle(iEvent, objectsToken_);

  std::vector<int> objectFlags;

  for(auto& object : *objects){

    if (GainSwitchTools::hasEBGainSwitchIn5x5(*object.superCluster(), &ebRecHits, topology_)) {
      objectFlags.emplace_back(1);
    } else {
      objectFlags.emplace_back(0);
    }      

  }

  std::unique_ptr<intMap> bRefMap(new intMap);
  intMap::Filler intMapFiller(*bRefMap);
  intMapFiller.insert(objects, objectFlags.begin(), objectFlags.end());
  intMapFiller.fill();
  iEvent.put(std::move(bRefMap),hasGainSwitchFlag_);

}

template<typename C>
void EGGainSwitchFlagProducer<C>::beginLuminosityBlock(edm::LuminosityBlock const& lb,
							edm::EventSetup const& es) {
  edm::ESHandle<CaloTopology> caloTopo ;
  es.get<CaloTopologyRecord>().get(caloTopo);
  topology_ = caloTopo.product();
}

using PhotonGainSwitchFlagProducer=EGGainSwitchFlagProducer<reco::PhotonCollection>;
using ElectronGainSwitchFlagProducer=EGGainSwitchFlagProducer<reco::GsfElectronCollection>;
//using PhotonGainSwitchFlagProducer=EGGainSwitchFlagProducer<edm::View<pat::Photon > >;
//using ElectronGainSwitchFlagProducer=EGGainSwitchFlagProducer<edm::View<pat::Electron > >;

DEFINE_FWK_MODULE(PhotonGainSwitchFlagProducer);
DEFINE_FWK_MODULE(ElectronGainSwitchFlagProducer);
