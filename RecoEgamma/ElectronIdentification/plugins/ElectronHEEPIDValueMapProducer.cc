#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/View.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"

#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/ElectronTkIsolation.h"

#include <memory>
#include <vector>

//Heavily inspired from ElectronIDValueMapProducer


class ElectronHEEPIDValueMapProducer : public edm::stream::EDProducer<> {

  public:
  
  explicit ElectronHEEPIDValueMapProducer(const edm::ParameterSet&);
  ~ElectronHEEPIDValueMapProducer();
  
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  
  private:
  
  virtual void produce(edm::Event&, const edm::EventSetup&) override;

  template<typename T>
  static void writeValueMap(edm::Event &iEvent,
			    const edm::Handle<edm::View<reco::GsfElectron> > & handle,
			    const std::vector<T> & values,
			    const std::string& label);
  
  static int nrSaturatedCrysIn5x5(const reco::GsfElectron& ele,
				  edm::Handle<EcalRecHitCollection>& ebHits,
				  edm::Handle<EcalRecHitCollection>& eeHits,
				  edm::ESHandle<CaloTopology>& caloTopo);

  template <typename T> void setToken(edm::EDGetTokenT<T>& token,edm::InputTag tag){token=consumes<T>(tag);}
  template <typename T> void setToken(edm::EDGetTokenT<T>& token,const edm::ParameterSet& iPara,const std::string& tag){token=consumes<T>(iPara.getParameter<edm::InputTag>(tag));}
  template<typename T> edm::Handle<T> getHandle(const edm::Event& iEvent,const edm::EDGetTokenT<T>& token){
    edm::Handle<T> handle;
    iEvent.getByToken(token,handle);
    return handle;
  }
  

  edm::EDGetTokenT<EcalRecHitCollection> ebRecHitToken_;
  edm::EDGetTokenT<EcalRecHitCollection> eeRecHitToken_;
  edm::EDGetTokenT<edm::View<reco::GsfElectron> > eleToken_;
  edm::EDGetTokenT<reco::BeamSpot> beamSpotToken_;
  edm::EDGetTokenT<reco::TrackCollection> trkToken_;

  struct TrkIsoParam {
    double extRadius;
    double intRadiusBarrel;
    double intRadiusEndcap;
    double stripBarrel;
    double stripEndcap;
    double ptMin;
    double maxVtxDist;
    double drb;
    TrkIsoParam(const edm::ParameterSet& iPara);
  };
  const TrkIsoParam trkIsoParam_;
  static const std::string eleTrkPtIsoNoJetCoreLabel_;
  static const std::string eleNrSaturateIn5x5Label_;
};

const std::string ElectronHEEPIDValueMapProducer::eleTrkPtIsoNoJetCoreLabel_="eleTrkPtIsoNoJetCore";
const std::string ElectronHEEPIDValueMapProducer::eleNrSaturateIn5x5Label_="eleNrSaturateIn5x5";
 
ElectronHEEPIDValueMapProducer::TrkIsoParam::TrkIsoParam(const edm::ParameterSet& iParam):
  extRadius(iParam.getParameter<double>("extRadius")),
  intRadiusBarrel(iParam.getParameter<double>("intRadiusBarrel")),
  intRadiusEndcap(iParam.getParameter<double>("intRadiusEndcap")),
  stripBarrel(iParam.getParameter<double>("stripBarrel")),
  stripEndcap(iParam.getParameter<double>("stripEndcap")),
  ptMin(iParam.getParameter<double>("ptMin")),
  maxVtxDist(iParam.getParameter<double>("maxVtxDist")),
  drb(iParam.getParameter<double>("drb"))
{ 

}
  

ElectronHEEPIDValueMapProducer::ElectronHEEPIDValueMapProducer(const edm::ParameterSet& iConfig):
  trkIsoParam_(iConfig.getParameter<edm::ParameterSet>("trkIsoConfig"))
{
  setToken(ebRecHitToken_,iConfig,"ebRecHits");
  setToken(eeRecHitToken_,iConfig,"eeRecHits");
  setToken(eleToken_,iConfig,"eles");
  setToken(trkToken_,iConfig,"tracks");
  setToken(beamSpotToken_,iConfig,"beamSpot");
  
  produces<edm::ValueMap<float> >(eleTrkPtIsoNoJetCoreLabel_);  
  produces<edm::ValueMap<int> >(eleNrSaturateIn5x5Label_);  
}

ElectronHEEPIDValueMapProducer::~ElectronHEEPIDValueMapProducer()
{

}

void ElectronHEEPIDValueMapProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  auto eleHandle = getHandle(iEvent,eleToken_);
  auto trkHandle = getHandle(iEvent,trkToken_);
  auto ebRecHitHandle = getHandle(iEvent,ebRecHitToken_);
  auto eeRecHitHandle = getHandle(iEvent,eeRecHitToken_);
  auto beamSpotHandle = getHandle(iEvent,beamSpotToken_);
  
  edm::ESHandle<CaloTopology> caloTopoHandle;
  iSetup.get<CaloTopologyRecord>().get(caloTopoHandle);
  
  ElectronTkIsolation isolCorr(trkIsoParam_.extRadius,trkIsoParam_.intRadiusBarrel,trkIsoParam_.intRadiusEndcap,
			       trkIsoParam_.stripBarrel,trkIsoParam_.stripEndcap,trkIsoParam_.ptMin,
			       trkIsoParam_.maxVtxDist,trkIsoParam_.drb,
			       trkHandle.product(),beamSpotHandle->position());
  isolCorr.setAlgosToReject({reco::TrackBase::jetCoreRegionalStep});

  std::vector<float> eleTrkPtIsoNoJetCore;
  std::vector<int> eleNrSaturateIn5x5;

  for(size_t eleNr=0;eleNr<eleHandle->size();eleNr++){
    auto elePtr = eleHandle->ptrAt(eleNr);
    eleTrkPtIsoNoJetCore.push_back(isolCorr.getPtTracks(&*elePtr));
    eleNrSaturateIn5x5.push_back(nrSaturatedCrysIn5x5(*elePtr,ebRecHitHandle,eeRecHitHandle,caloTopoHandle));    
  }
  
  writeValueMap(iEvent,eleHandle,eleTrkPtIsoNoJetCore,eleTrkPtIsoNoJetCoreLabel_);  
  writeValueMap(iEvent,eleHandle,eleNrSaturateIn5x5,eleNrSaturateIn5x5Label_);  
  
}

int ElectronHEEPIDValueMapProducer::nrSaturatedCrysIn5x5(const reco::GsfElectron& ele,
							 edm::Handle<EcalRecHitCollection>& ebHits,
							 edm::Handle<EcalRecHitCollection>& eeHits,
							 edm::ESHandle<CaloTopology>& caloTopo)
{ 
  DetId id = ele.superCluster()->seed()->seed();
  auto recHits = id.subdetId()==EcalBarrel ? ebHits.product() : eeHits.product();
  return noZS::EcalClusterTools::nrSaturatedCrysIn5x5(id,recHits,caloTopo.product());

}

template<typename T>
void ElectronHEEPIDValueMapProducer::writeValueMap(edm::Event &iEvent,
						   const edm::Handle<edm::View<reco::GsfElectron> > & handle,
						   const std::vector<T> & values,
						   const std::string& label)
{ 
  std::unique_ptr<edm::ValueMap<T> > valMap(new edm::ValueMap<T>());
  typename edm::ValueMap<T>::Filler filler(*valMap);
  filler.insert(handle, values.begin(), values.end());
  filler.fill();
  iEvent.put(std::move(valMap),label);
}

void ElectronHEEPIDValueMapProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(ElectronHEEPIDValueMapProducer);
