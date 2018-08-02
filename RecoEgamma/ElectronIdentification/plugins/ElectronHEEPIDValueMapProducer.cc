#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"

#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"

#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EleTkIsolFromCands.h"

#include "RecoEgamma/EgammaTools/interface/MultiToken.h"
#include "RecoEgamma/EgammaTools/interface/Utils.h"

#include <vector>

//Heavily inspired from ElectronIDValueMapProducer


class ElectronHEEPIDValueMapProducer : public edm::stream::EDProducer<> {

public:
  explicit ElectronHEEPIDValueMapProducer(const edm::ParameterSet&);
  ~ElectronHEEPIDValueMapProducer() override {};

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  static int nrSaturatedCrysIn5x5(const reco::GsfElectron& ele,
				  const edm::Handle<EcalRecHitCollection>& ebHits,
				  const edm::Handle<EcalRecHitCollection>& eeHits,
				  const edm::ESHandle<CaloTopology>& caloTopo);

  float calTrkIso(const reco::GsfElectron& ele,
		  const edm::View<reco::GsfElectron>& eles,
		  const std::vector<edm::Handle<pat::PackedCandidateCollection> >& handles,
		  const std::vector<EleTkIsolFromCands::PIDVeto>& pidVetos)const;

  template<typename T>
  static std::vector<edm::Handle<T> >
  getHandles(const edm::Event& iEvent,
             const std::vector<edm::EDGetTokenT<T>>& tokens){
    std::vector<edm::Handle<T> > handles(tokens.size());
    for(size_t i = 1; i < tokens.size(); ++i){
      iEvent.getByToken(tokens[i],handles[i]);
    }
    return handles;
  }

  int dataFormat_;

  MultiToken<EcalRecHitCollection> ebRecHitToken_;
  MultiToken<EcalRecHitCollection> eeRecHitToken_;
  MultiToken<edm::View<reco::GsfElectron> > eleToken_;
  std::vector<edm::EDGetTokenT<pat::PackedCandidateCollection> >candTokensAOD_;
  std::vector<edm::EDGetTokenT<pat::PackedCandidateCollection> >candTokensMiniAOD_;
  MultiToken<reco::BeamSpot> beamSpotToken_;

  EleTkIsolFromCands trkIsoCalc_;
  std::vector<EleTkIsolFromCands::PIDVeto> candVetosAOD_;
  std::vector<EleTkIsolFromCands::PIDVeto> candVetosMiniAOD_;

  static const std::string eleTrkPtIsoLabel_;
  static const std::string eleNrSaturateIn5x5Label_;
};

const std::string ElectronHEEPIDValueMapProducer::eleTrkPtIsoLabel_="eleTrkPtIso";
const std::string ElectronHEEPIDValueMapProducer::eleNrSaturateIn5x5Label_="eleNrSaturateIn5x5";



ElectronHEEPIDValueMapProducer::ElectronHEEPIDValueMapProducer(const edm::ParameterSet& iConfig)
  : dataFormat_(iConfig.getParameter<int>("dataFormat"))
  , ebRecHitToken_(
        consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("ebRecHitsAOD")),
        consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("ebRecHitsMiniAOD")))
  , eeRecHitToken_(
        consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("eeRecHitsAOD")),
        consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("eeRecHitsMiniAOD")))
  , eleToken_(
        consumes<edm::View<reco::GsfElectron>>(iConfig.getParameter<edm::InputTag>("elesAOD")),
        consumes<edm::View<reco::GsfElectron>>(iConfig.getParameter<edm::InputTag>("elesMiniAOD")))
  , beamSpotToken_(consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamSpot")))
  , trkIsoCalc_(iConfig.getParameter<edm::ParameterSet>("trkIsoConfig"))
{

  // Set the token indices for the MultiTokens, so they don't need to figure it
  // out themselves in case we already know.
  ebRecHitToken_.setGoodTokenIndex(dataFormat_ - 1);
  eeRecHitToken_.setGoodTokenIndex(dataFormat_ - 1);
  eleToken_.setGoodTokenIndex(dataFormat_ - 1);
  beamSpotToken_.setGoodTokenIndex(0);

  for(auto tag: iConfig.getParameter<std::vector<edm::InputTag>>("candsAOD"))
      candTokensAOD_.push_back(consumes<pat::PackedCandidateCollection>(tag));
  for(auto tag: iConfig.getParameter<std::vector<edm::InputTag>>("candsMiniAOD"))
      candTokensMiniAOD_.push_back(consumes<pat::PackedCandidateCollection>(tag));

  auto fillVetos=[](const auto& in,auto& out){
    std::transform(in.begin(),in.end(),std::back_inserter(out),EleTkIsolFromCands::pidVetoFromStr);
  };

  fillVetos(iConfig.getParameter<std::vector<std::string> >("candVetosAOD"),candVetosAOD_);
  if(candVetosAOD_.size()!=iConfig.getParameter<std::vector<edm::InputTag> >("candsAOD").size()){
    throw cms::Exception("ConfigError") <<" Error candVetosAOD should be the same size as candsAOD "<<std::endl;
  }

  fillVetos(iConfig.getParameter<std::vector<std::string> >("candVetosMiniAOD"),candVetosMiniAOD_);
  if(candVetosMiniAOD_.size()!=iConfig.getParameter<std::vector<edm::InputTag> >("candsMiniAOD").size()){
    throw cms::Exception("ConfigError") <<" Error candVetosMiniAOD should be the same size as candsMiniAOD "<<std::endl;
  }

  produces<edm::ValueMap<float> >(eleTrkPtIsoLabel_);
  produces<edm::ValueMap<int> >(eleNrSaturateIn5x5Label_);
}

void ElectronHEEPIDValueMapProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  auto eleHandle      = eleToken_.getValidHandle(iEvent);
  if (dataFormat_ == 0) {
      // auto detect
      dataFormat_ = eleToken_.getGoodTokenIndex() + 1;
  }
  auto ebRecHitHandle = ebRecHitToken_.getValidHandle(iEvent);
  auto eeRecHitHandle = eeRecHitToken_.getValidHandle(iEvent);
  auto beamSpotHandle = beamSpotToken_.getValidHandle(iEvent);
  auto candHandles    = dataFormat_ == 1 ? getHandles(iEvent,candTokensAOD_) : getHandles(iEvent,candTokensMiniAOD_);

  const auto& candVetos = dataFormat_ == 1 ? candVetosAOD_ : candVetosMiniAOD_;

  edm::ESHandle<CaloTopology> caloTopoHandle;
  iSetup.get<CaloTopologyRecord>().get(caloTopoHandle);

  std::vector<float> eleTrkPtIso;
  std::vector<int> eleNrSaturateIn5x5;
  for(size_t eleNr=0;eleNr<eleHandle->size();eleNr++){
    auto elePtr = eleHandle->ptrAt(eleNr);
    eleTrkPtIso.push_back(calTrkIso(*elePtr,*eleHandle,candHandles,candVetos));
    eleNrSaturateIn5x5.push_back(nrSaturatedCrysIn5x5(*elePtr,ebRecHitHandle,eeRecHitHandle,caloTopoHandle));
  }

  writeValueMap(iEvent,eleHandle,eleTrkPtIso,eleTrkPtIsoLabel_);
  writeValueMap(iEvent,eleHandle,eleNrSaturateIn5x5,eleNrSaturateIn5x5Label_);
}

int ElectronHEEPIDValueMapProducer::nrSaturatedCrysIn5x5(const reco::GsfElectron& ele,
							 const edm::Handle<EcalRecHitCollection>& ebHits,
							 const edm::Handle<EcalRecHitCollection>& eeHits,
							 const edm::ESHandle<CaloTopology>& caloTopo)
{
  DetId id = ele.superCluster()->seed()->seed();
  auto recHits = id.subdetId()==EcalBarrel ? ebHits.product() : eeHits.product();
  return noZS::EcalClusterTools::nrSaturatedCrysIn5x5(id,recHits,caloTopo.product());

}

float ElectronHEEPIDValueMapProducer::
calTrkIso(const reco::GsfElectron& ele,
	  const edm::View<reco::GsfElectron>& eles,
	  const std::vector<edm::Handle<pat::PackedCandidateCollection> >& handles,
	  const std::vector<EleTkIsolFromCands::PIDVeto>& pidVetos)const
{
  if(ele.gsfTrack().isNull()) return std::numeric_limits<float>::max();
  else{
    float trkIso=0.;
    for(size_t handleNr=0;handleNr<handles.size();handleNr++){
      auto& handle = handles[handleNr];
      if(handle.isValid()){
	if(handleNr<pidVetos.size()){
	  trkIso+= trkIsoCalc_.calIsolPt(*ele.gsfTrack(),*handle,pidVetos[handleNr]);
	}else{
	  throw cms::Exception("LogicError") <<" somehow the pidVetos and handles do not much, given this is checked at construction time, something has gone wrong in the code handle nr "<<handleNr<<" size of vetos "<<pidVetos.size();
	}
      }
    }
    return trkIso;
  }
}

void ElectronHEEPIDValueMapProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {

  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("beamSpot",edm::InputTag("offlineBeamSpot"));
  desc.add<edm::InputTag>("ebRecHitsAOD",edm::InputTag("reducedEcalRecHitsEB"));
  desc.add<edm::InputTag>("eeRecHitsAOD",edm::InputTag("reducedEcalRecHitsEE"));
  desc.add<std::vector<edm::InputTag> >("candsAOD",{edm::InputTag("packedCandidates")});
  desc.add<std::vector<std::string> >("candVetosAOD",{"none"});
  desc.add<edm::InputTag>("elesAOD",edm::InputTag("gedGsfElectrons"));

  desc.add<edm::InputTag>("ebRecHitsMiniAOD",edm::InputTag("reducedEcalRecHitsEB"));
  desc.add<edm::InputTag>("eeRecHitsMiniAOD",edm::InputTag("reducedEcalRecHitsEE"));
  desc.add<std::vector<edm::InputTag> >("candsMiniAOD",{edm::InputTag("packedCandidates")});
  desc.add<std::vector<std::string> >("candVetosMiniAOD",{"none"});
  desc.add<edm::InputTag>("elesMiniAOD",edm::InputTag("gedGsfElectrons"));
  desc.add<int>("dataFormat",0);

  desc.add("trkIsoConfig",EleTkIsolFromCands::pSetDescript());

  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(ElectronHEEPIDValueMapProducer);
