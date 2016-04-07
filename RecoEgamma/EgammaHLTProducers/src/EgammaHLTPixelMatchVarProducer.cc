

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateIsolation.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"

#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"

#include "RecoEgamma/EgammaHLTProducers/interface/EgammaHLTPixelMatchParamObjects.h"

class EgammaHLTPixelMatchVarProducer : public edm::global::EDProducer<> {
public:

  explicit EgammaHLTPixelMatchVarProducer(const edm::ParameterSet&);
  ~EgammaHLTPixelMatchVarProducer();
  
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  void produce(edm::StreamID sid, edm::Event&, const edm::EventSetup&) const override;
  std::array<float,4> calS2(const reco::ElectronSeed& seed,int charge)const;

private: 
  // ----------member data ---------------------------
  
  const edm::EDGetTokenT<reco::RecoEcalCandidateCollection> recoEcalCandidateToken_;
  const edm::EDGetTokenT<reco::ElectronSeedCollection> pixelSeedsToken_;

  egPM::Param<reco::ElectronSeed> dPhi1Para_;
  egPM::Param<reco::ElectronSeed> dPhi2Para_;
  egPM::Param<reco::ElectronSeed> dRZ2Para_;
  
  int productsToWrite_;
  
};

EgammaHLTPixelMatchVarProducer::EgammaHLTPixelMatchVarProducer(const edm::ParameterSet& config) : 
  recoEcalCandidateToken_(consumes<reco::RecoEcalCandidateCollection>(config.getParameter<edm::InputTag>("recoEcalCandidateProducer"))),
  pixelSeedsToken_(consumes<reco::ElectronSeedCollection>(config.getParameter<edm::InputTag>("pixelSeedsProducer"))),
  dPhi1Para_(config.getParameter<edm::ParameterSet>("dPhi1SParams")),
  dPhi2Para_(config.getParameter<edm::ParameterSet>("dPhi2SParams")),
  dRZ2Para_(config.getParameter<edm::ParameterSet>("dRZ2SParams")),
  productsToWrite_(config.getParameter<int>("productsToWrite"))
  
{
  //register your products  
  produces < reco::RecoEcalCandidateIsolationMap >("s2");
  if(productsToWrite_>=1){
    produces < reco::RecoEcalCandidateIsolationMap >("dPhi1BestS2");
    produces < reco::RecoEcalCandidateIsolationMap >("dPhi2BestS2");
    produces < reco::RecoEcalCandidateIsolationMap >("dzBestS2");
  }


}

EgammaHLTPixelMatchVarProducer::~EgammaHLTPixelMatchVarProducer()
{}

void EgammaHLTPixelMatchVarProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {

  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>(("recoEcalCandidateProducer"), edm::InputTag("hltL1SeededRecoEcalCandidate"));
  desc.add<edm::InputTag>(("pixelSeedsProducer"), edm::InputTag("electronPixelSeeds"));
  
  edm::ParameterSetDescription varParamDesc;
  edm::ParameterSetDescription binParamDesc;
  
  std::auto_ptr<edm::ParameterDescriptionCases<std::string>> binDescCases;
  binDescCases = 
    "AbsEtaClus" >> 
    (edm::ParameterDescription<double>("xMin",0.0,true) and
     edm::ParameterDescription<double>("xMax",3.0,true) and
     edm::ParameterDescription<int>("yMin",0,true) and
     edm::ParameterDescription<int>("yMax",99999,true) and
     edm::ParameterDescription<std::string>("funcType","pol0",true) and
     edm::ParameterDescription<std::vector<double>>("funcParams",{0.},true)) or
    "AbsEtaClusPhi" >>
    (edm::ParameterDescription<double>("xMin",0.0,true) and
     edm::ParameterDescription<double>("xMax",3.0,true) and
     edm::ParameterDescription<int>("yMin",0,true) and
     edm::ParameterDescription<int>("yMax",99999,true) and
     edm::ParameterDescription<std::string>("funcType","pol0",true) and
     edm::ParameterDescription<std::vector<double>>("funcParams",{0.},true)) or 
     "AbsEtaClusEt" >>
    (edm::ParameterDescription<double>("xMin",0.0,true) and
     edm::ParameterDescription<double>("xMax",3.0,true) and
     edm::ParameterDescription<int>("yMin",0,true) and
     edm::ParameterDescription<int>("yMax",99999,true) and
     edm::ParameterDescription<std::string>("funcType","pol0",true) and
     edm::ParameterDescription<std::vector<double>>("funcParams",{0.},true));
  
  binParamDesc.ifValue(edm::ParameterDescription<std::string>("binType","AbsEtaClus",true), binDescCases);
  
  
  varParamDesc.addVPSet("bins",binParamDesc);
  desc.add("dPhi1SParams",varParamDesc);
  desc.add("dPhi2SParams",varParamDesc);
  desc.add("dRZ2SParams",varParamDesc);
  desc.add<int>("productsToWrite",0);
  descriptions.add(("hltEgammaHLTPixelMatchVarProducer"), desc);  
}

void EgammaHLTPixelMatchVarProducer::produce(edm::StreamID sid, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  
  // Get the HLT filtered objects
  edm::Handle<reco::RecoEcalCandidateCollection> recoEcalCandHandle;
  iEvent.getByToken(recoEcalCandidateToken_,recoEcalCandHandle);


  edm::Handle<reco::ElectronSeedCollection> pixelSeedsHandle;
  iEvent.getByToken(pixelSeedsToken_,pixelSeedsHandle);

  if(!recoEcalCandHandle.isValid() || !pixelSeedsHandle.isValid()) return;

  std::auto_ptr<reco::RecoEcalCandidateIsolationMap> dPhi1BestS2Map(new reco::RecoEcalCandidateIsolationMap(recoEcalCandHandle));
  std::auto_ptr<reco::RecoEcalCandidateIsolationMap> dPhi2BestS2Map(new reco::RecoEcalCandidateIsolationMap(recoEcalCandHandle));
  std::auto_ptr<reco::RecoEcalCandidateIsolationMap> dzBestS2Map(new reco::RecoEcalCandidateIsolationMap(recoEcalCandHandle));
  std::auto_ptr<reco::RecoEcalCandidateIsolationMap> s2Map(new reco::RecoEcalCandidateIsolationMap(recoEcalCandHandle));
  
  for(unsigned int candNr = 0; candNr<recoEcalCandHandle->size(); candNr++) {
    
    reco::RecoEcalCandidateRef candRef(recoEcalCandHandle,candNr);
    reco::SuperClusterRef candSCRef = candRef->superCluster();
    
    std::array<float,4> bestS2{{std::numeric_limits<float>::max(),std::numeric_limits<float>::max(),std::numeric_limits<float>::max(),std::numeric_limits<float>::max()}};
    for(auto & seed : *pixelSeedsHandle){
      edm::RefToBase<reco::CaloCluster> pixelClusterRef = seed.caloCluster() ;
      reco::SuperClusterRef pixelSCRef = pixelClusterRef.castTo<reco::SuperClusterRef>() ;
      if(&(*candSCRef) ==  &(*pixelSCRef)){
	
	std::array<float,4> s2Data = calS2(seed,-1);
	std::array<float,4> s2DataPos = calS2(seed,+1);
	if(s2Data[0]<bestS2[0]) bestS2=s2Data;
	if(s2DataPos[0]<bestS2[0]) bestS2=s2DataPos;
	
      }
    }

   
    s2Map->insert(candRef,bestS2[0]);
    if(productsToWrite_>=1){
      dPhi1BestS2Map->insert(candRef,bestS2[1]);
      dPhi2BestS2Map->insert(candRef,bestS2[2]);
      dzBestS2Map->insert(candRef,bestS2[3]);
    }
    
  }

  iEvent.put(s2Map,"s2");
  if(productsToWrite_>=1){
    iEvent.put(dPhi1BestS2Map,"dPhi1BestS2");
    iEvent.put(dPhi2BestS2Map,"dPhi2BestS2");
    iEvent.put(dzBestS2Map,"dzBestS2");
  }
}

std::array<float,4> EgammaHLTPixelMatchVarProducer::calS2(const reco::ElectronSeed& seed,int charge)const
{
  const float dPhi1Const = dPhi1Para_(seed);	
  const float dPhi2Const = dPhi2Para_(seed);
  const float dRZ2Const = dRZ2Para_(seed);
  
  float dPhi1 = (charge <0 ? seed.dPhi1() : seed.dPhi1Pos())/dPhi1Const;
  float dPhi2 = (charge <0 ? seed.dPhi2() : seed.dPhi2Pos())/dPhi2Const;
  float dRz2 = (charge <0 ? seed.dRz2() : seed.dRz2Pos())/dRZ2Const;
  
  float s2 = dPhi1*dPhi1+dPhi2*dPhi2+dRz2*dRz2;
  return std::array<float,4>{{s2,dPhi1,dPhi2,dRz2}}; 
}


DEFINE_FWK_MODULE(EgammaHLTPixelMatchVarProducer);
