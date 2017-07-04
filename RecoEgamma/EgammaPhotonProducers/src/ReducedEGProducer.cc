#include <iostream>
#include <vector>
#include <memory>
#include <unordered_set>

// Framework
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "CommonTools/Utils/interface/StringToEnumValue.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"


#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/EgammaReco/interface/ClusterShape.h"
#include "DataFormats/EgammaCandidates/interface/PhotonCore.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateEGammaExtra.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateEGammaExtraFwd.h"


#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "RecoCaloTools/Selectors/interface/CaloConeSelector.h"

#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgoRcd.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"

#include "RecoEgamma/EgammaPhotonProducers/interface/ReducedEGProducer.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaTowerIsolation.h"

#include "RecoEcal/EgammaCoreTools/interface/EcalClusterFunctionBaseClass.h" 
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterFunctionFactory.h" 
#include "RecoEcal/EgammaCoreTools/plugins/EcalClusterCrackCorrection.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaHadTower.h"

#include "RecoEgamma/EgammaTools/interface/ConversionTools.h"

namespace std {
  template<> 
  struct hash<DetId> {
    size_t operator()(const DetId& id) const {
      return std::hash<uint32_t>()(id.rawId());
    }
  };  
}

ReducedEGProducer::ReducedEGProducer(const edm::ParameterSet& config) :
  photonT_(consumes<reco::PhotonCollection>(config.getParameter<edm::InputTag>("photons"))),
  ootPhotonT_(consumes<reco::PhotonCollection>(config.getParameter<edm::InputTag>("ootPhotons"))),
  gsfElectronT_(consumes<reco::GsfElectronCollection>(config.getParameter<edm::InputTag>("gsfElectrons"))),
  conversionT_(consumes<reco::ConversionCollection>(config.getParameter<edm::InputTag>("conversions"))),
  singleConversionT_(consumes<reco::ConversionCollection>(config.getParameter<edm::InputTag>("singleConversions"))),
  barrelEcalHits_(consumes<EcalRecHitCollection>(config.getParameter<edm::InputTag>("barrelEcalHits"))),
  endcapEcalHits_(consumes<EcalRecHitCollection>(config.getParameter<edm::InputTag>("endcapEcalHits"))),
  doPreshowerEcalHits_(!config.getParameter<edm::InputTag>("preshowerEcalHits").label().empty()),
  preshowerEcalHits_(doPreshowerEcalHits_ ? consumes<EcalRecHitCollection>(config.getParameter<edm::InputTag>("preshowerEcalHits")) : edm::EDGetTokenT<EcalRecHitCollection>()),
  photonPfCandMapT_(consumes<edm::ValueMap<std::vector<reco::PFCandidateRef> > >(config.getParameter<edm::InputTag>("photonsPFValMap"))),  
  gsfElectronPfCandMapT_(consumes<edm::ValueMap<std::vector<reco::PFCandidateRef> > >(config.getParameter<edm::InputTag>("gsfElectronsPFValMap"))),
  //output collections    
  outPhotons_("reducedGedPhotons"),
  outPhotonCores_("reducedGedPhotonCores"),
  outOOTPhotons_("reducedOOTPhotons"),
  outOOTPhotonCores_("reducedOOTPhotonCores"),
  outGsfElectrons_("reducedGedGsfElectrons"),
  outGsfElectronCores_("reducedGedGsfElectronCores"),
  outConversions_("reducedConversions"),
  outSingleConversions_("reducedSingleLegConversions"),
  outSuperClusters_("reducedSuperClusters"),
  outEBEEClusters_("reducedEBEEClusters"),
  outESClusters_("reducedESClusters"),
  outOOTSuperClusters_("reducedOOTSuperClusters"),
  outOOTEBEEClusters_("reducedOOTEBEEClusters"),
  outOOTESClusters_("reducedOOTESClusters"),
  outEBRecHits_("reducedEBRecHits"),
  outEERecHits_("reducedEERecHits"),
  outESRecHits_("reducedESRecHits"),
  outPhotonPfCandMap_("reducedPhotonPfCandMap"),
  outGsfElectronPfCandMap_("reducedGsfElectronPfCandMap"),
  outPhotonIds_(config.getParameter<std::vector<std::string> >("photonIDOutput")),
  outGsfElectronIds_(config.getParameter<std::vector<std::string> >("gsfElectronIDOutput")),
  outPhotonPFClusterIsos_(config.getParameter<std::vector<std::string> >("photonPFClusterIsoOutput")),
  outOOTPhotonPFClusterIsos_(config.getParameter<std::vector<std::string> >("ootPhotonPFClusterIsoOutput")),
  outGsfElectronPFClusterIsos_(config.getParameter<std::vector<std::string> >("gsfElectronPFClusterIsoOutput")),
  keepPhotonSel_(config.getParameter<std::string>("keepPhotons")),
  slimRelinkPhotonSel_(config.getParameter<std::string>("slimRelinkPhotons")),
  relinkPhotonSel_(config.getParameter<std::string>("relinkPhotons")),
  keepOOTPhotonSel_(config.getParameter<std::string>("keepOOTPhotons")),
  slimRelinkOOTPhotonSel_(config.getParameter<std::string>("slimRelinkOOTPhotons")),
  relinkOOTPhotonSel_(config.getParameter<std::string>("relinkOOTPhotons")),
  keepGsfElectronSel_(config.getParameter<std::string>("keepGsfElectrons")),
  slimRelinkGsfElectronSel_(config.getParameter<std::string>("slimRelinkGsfElectrons")),
  relinkGsfElectronSel_(config.getParameter<std::string>("relinkGsfElectrons"))
{  
  const std::vector<edm::InputTag>& photonidinputs = 
    config.getParameter<std::vector<edm::InputTag> >("photonIDSources");
  for (const edm::InputTag &tag : photonidinputs) {
    photonIdTs_.emplace_back(consumes<edm::ValueMap<bool> >(tag));
  }
  
  const std::vector<edm::InputTag>& gsfelectronidinputs = 
    config.getParameter<std::vector<edm::InputTag> >("gsfElectronIDSources");
  for (const edm::InputTag &tag : gsfelectronidinputs) {
    gsfElectronIdTs_.emplace_back(consumes<edm::ValueMap<float> >(tag));
  }  
  
  const std::vector<edm::InputTag>&  photonpfclusterisoinputs = 
    config.getParameter<std::vector<edm::InputTag> >("photonPFClusterIsoSources");
  for (const edm::InputTag &tag : photonpfclusterisoinputs) {
    photonPFClusterIsoTs_.emplace_back(consumes<edm::ValueMap<float> >(tag));
  }  

  const std::vector<edm::InputTag>&  ootphotonpfclusterisoinputs = 
    config.getParameter<std::vector<edm::InputTag> >("ootPhotonPFClusterIsoSources");
  for (const edm::InputTag &tag : ootphotonpfclusterisoinputs) {
    ootPhotonPFClusterIsoTs_.emplace_back(consumes<edm::ValueMap<float> >(tag));
  }  

  const std::vector<edm::InputTag>& gsfelectronpfclusterisoinputs = 
    config.getParameter<std::vector<edm::InputTag> >("gsfElectronPFClusterIsoSources");
  for (const edm::InputTag &tag : gsfelectronpfclusterisoinputs) {
    gsfElectronPFClusterIsoTs_.emplace_back(consumes<edm::ValueMap<float> >(tag));
  }  
  
  produces< reco::PhotonCollection >(outPhotons_);
  produces< reco::PhotonCoreCollection >(outPhotonCores_);
  produces< reco::PhotonCollection >(outOOTPhotons_);
  produces< reco::PhotonCoreCollection >(outOOTPhotonCores_);
  produces< reco::GsfElectronCollection >(outGsfElectrons_);
  produces< reco::GsfElectronCoreCollection >(outGsfElectronCores_);
  produces< reco::ConversionCollection >(outConversions_);
  produces< reco::ConversionCollection >(outSingleConversions_);
  produces< reco::SuperClusterCollection >(outSuperClusters_);  
  produces< reco::CaloClusterCollection >(outEBEEClusters_);
  produces< reco::CaloClusterCollection >(outESClusters_);
  produces< reco::SuperClusterCollection >(outOOTSuperClusters_);  
  produces< reco::CaloClusterCollection >(outOOTEBEEClusters_);
  produces< reco::CaloClusterCollection >(outOOTESClusters_);
  produces< EcalRecHitCollection >(outEBRecHits_);
  produces< EcalRecHitCollection >(outEERecHits_);
  if (doPreshowerEcalHits_) produces< EcalRecHitCollection >(outESRecHits_);    
  produces< edm::ValueMap<std::vector<reco::PFCandidateRef> > >(outPhotonPfCandMap_);    
  produces< edm::ValueMap<std::vector<reco::PFCandidateRef> > >(outGsfElectronPfCandMap_);   
  for (const std::string &outid : outPhotonIds_) {
    produces< edm::ValueMap<bool> >(outid);   
  }
  for (const std::string &outid : outGsfElectronIds_) {
    produces< edm::ValueMap<float> >(outid);   
  }  
  for (const std::string &outid : outPhotonPFClusterIsos_) {
    produces< edm::ValueMap<float> >(outid);   
  }
  for (const std::string &outid : outOOTPhotonPFClusterIsos_) {
    produces< edm::ValueMap<float> >(outid);   
  }
  for (const std::string &outid : outGsfElectronPFClusterIsos_) {
    produces< edm::ValueMap<float> >(outid);   
  }
}

ReducedEGProducer::~ReducedEGProducer() 
{
}

void ReducedEGProducer::produce(edm::Event& theEvent, const edm::EventSetup& theEventSetup) {

  //get input collections
  
  
  edm::Handle<reco::PhotonCollection> photonHandle;
  theEvent.getByToken(photonT_, photonHandle);

  edm::Handle<reco::PhotonCollection> ootPhotonHandle;
  theEvent.getByToken(ootPhotonT_, ootPhotonHandle);

  edm::Handle<reco::GsfElectronCollection> gsfElectronHandle;
  theEvent.getByToken(gsfElectronT_, gsfElectronHandle);  
  
  edm::Handle<reco::ConversionCollection> conversionHandle;
  theEvent.getByToken(conversionT_, conversionHandle);  

  edm::Handle<reco::ConversionCollection> singleConversionHandle;
  theEvent.getByToken(singleConversionT_, singleConversionHandle);    
  
  edm::Handle<EcalRecHitCollection> barrelHitHandle;
  theEvent.getByToken(barrelEcalHits_, barrelHitHandle);  
  
  edm::Handle<EcalRecHitCollection> endcapHitHandle;
  theEvent.getByToken(endcapEcalHits_, endcapHitHandle);

  edm::Handle<EcalRecHitCollection> preshowerHitHandle;
  if (doPreshowerEcalHits_) theEvent.getByToken(preshowerEcalHits_, preshowerHitHandle);
  
  edm::Handle<edm::ValueMap<std::vector<reco::PFCandidateRef> > > photonPfCandMapHandle;
  theEvent.getByToken(photonPfCandMapT_, photonPfCandMapHandle);  

  edm::Handle<edm::ValueMap<std::vector<reco::PFCandidateRef> > > gsfElectronPfCandMapHandle;
  theEvent.getByToken(gsfElectronPfCandMapT_, gsfElectronPfCandMapHandle);    
  
  std::vector<edm::Handle<edm::ValueMap<bool> > > photonIdHandles(photonIdTs_.size());
  for (unsigned int itok=0; itok<photonIdTs_.size(); ++itok) {
    theEvent.getByToken(photonIdTs_[itok],photonIdHandles[itok]);
  }
  
  std::vector<edm::Handle<edm::ValueMap<float> > > gsfElectronIdHandles(gsfElectronIdTs_.size());
  for (unsigned int itok=0; itok<gsfElectronIdTs_.size(); ++itok) {
    theEvent.getByToken(gsfElectronIdTs_[itok],gsfElectronIdHandles[itok]);
  }  
  
  std::vector<edm::Handle<edm::ValueMap<float> > > gsfElectronPFClusterIsoHandles(gsfElectronPFClusterIsoTs_.size());
  for (unsigned int itok=0; itok<gsfElectronPFClusterIsoTs_.size(); ++itok) {
    theEvent.getByToken(gsfElectronPFClusterIsoTs_[itok],gsfElectronPFClusterIsoHandles[itok]);
  }  
  
  std::vector<edm::Handle<edm::ValueMap<float> > > photonPFClusterIsoHandles(photonPFClusterIsoTs_.size());
  for (unsigned int itok=0; itok<photonPFClusterIsoTs_.size(); ++itok) {
    theEvent.getByToken(photonPFClusterIsoTs_[itok],photonPFClusterIsoHandles[itok]);
  }  
  
  std::vector<edm::Handle<edm::ValueMap<float> > > ootPhotonPFClusterIsoHandles(ootPhotonPFClusterIsoTs_.size());
  for (unsigned int itok=0; itok<ootPhotonPFClusterIsoTs_.size(); ++itok) {
    theEvent.getByToken(ootPhotonPFClusterIsoTs_[itok],ootPhotonPFClusterIsoHandles[itok]);
  }  
  
  edm::ESHandle<CaloTopology> theCaloTopology;
  theEventSetup.get<CaloTopologyRecord>().get(theCaloTopology);  
  const CaloTopology *caloTopology = & (*theCaloTopology);  
  
  //initialize output collections
  auto photons = std::make_unique<reco::PhotonCollection>();
  auto photonCores = std::make_unique<reco::PhotonCoreCollection>();
  auto ootPhotons = std::make_unique<reco::PhotonCollection>();
  auto ootPhotonCores = std::make_unique<reco::PhotonCoreCollection>();
  auto gsfElectrons = std::make_unique<reco::GsfElectronCollection>();
  auto gsfElectronCores = std::make_unique<reco::GsfElectronCoreCollection>();
  auto conversions = std::make_unique<reco::ConversionCollection>();
  auto singleConversions = std::make_unique<reco::ConversionCollection>();
  auto superClusters = std::make_unique<reco::SuperClusterCollection>();
  auto ebeeClusters = std::make_unique<reco::CaloClusterCollection>();
  auto esClusters = std::make_unique<reco::CaloClusterCollection>();
  auto ootSuperClusters = std::make_unique<reco::SuperClusterCollection>();
  auto ootEbeeClusters = std::make_unique<reco::CaloClusterCollection>();
  auto ootEsClusters = std::make_unique<reco::CaloClusterCollection>();
  auto ebRecHits = std::make_unique<EcalRecHitCollection>();
  auto eeRecHits = std::make_unique<EcalRecHitCollection>();
  auto esRecHits = std::make_unique<EcalRecHitCollection>();
  auto photonPfCandMap = std::make_unique<edm::ValueMap<std::vector<reco::PFCandidateRef>>>();
  auto gsfElectronPfCandMap = std::make_unique<edm::ValueMap<std::vector<reco::PFCandidateRef>>>();

  //maps to collection indices of output objects
  std::map<reco::PhotonCoreRef, unsigned int> photonCoreMap;
  std::map<reco::PhotonCoreRef, unsigned int> ootPhotonCoreMap;
  std::map<reco::GsfElectronCoreRef, unsigned int> gsfElectronCoreMap;
  std::map<reco::ConversionRef, unsigned int> conversionMap;
  std::map<reco::ConversionRef, unsigned int> singleConversionMap;
  std::map<reco::SuperClusterRef, unsigned int> superClusterMap;
  std::map<reco::CaloClusterPtr, unsigned int> ebeeClusterMap;
  std::map<reco::CaloClusterPtr, unsigned int> esClusterMap;
  std::map<reco::SuperClusterRef, unsigned int> ootSuperClusterMap;
  std::map<reco::CaloClusterPtr, unsigned int> ootEbeeClusterMap;
  std::map<reco::CaloClusterPtr, unsigned int> ootEsClusterMap;
  std::unordered_set<DetId> rechitMap;
  
  std::unordered_set<unsigned int> superClusterFullRelinkMap;
  std::unordered_set<unsigned int> ootSuperClusterFullRelinkMap;
  
  //vectors for pfcandidate valuemaps
  std::vector<std::vector<reco::PFCandidateRef> > pfCandIsoPairVecPho;  
  std::vector<std::vector<reco::PFCandidateRef> > pfCandIsoPairVecEle;
  
  //vectors for id valuemaps
  std::vector<std::vector<bool> > photonIdVals(photonIdHandles.size());
  std::vector<std::vector<float> > gsfElectronIdVals(gsfElectronIdHandles.size());
  std::vector<std::vector<float> > photonPFClusterIsoVals(photonPFClusterIsoHandles.size());
  std::vector<std::vector<float> > ootPhotonPFClusterIsoVals(ootPhotonPFClusterIsoHandles.size());
  std::vector<std::vector<float> > gsfElectronPFClusterIsoVals(gsfElectronPFClusterIsoHandles.size());
  
  //loop over photons and fill maps
  for (unsigned int ipho=0; ipho<photonHandle->size(); ++ipho) {
    const reco::Photon &photon = (*photonHandle)[ipho];
    
    bool keep = keepPhotonSel_(photon);
    if (!keep) continue;
    
    reco::PhotonRef photonref(photonHandle,ipho);
    
    photons->push_back(photon);
    
    //fill pf candidate value map vector
    pfCandIsoPairVecPho.push_back((*photonPfCandMapHandle)[photonref]);

    //fill photon id valuemap vectors
    for (unsigned int iid=0; iid<photonIdHandles.size(); ++iid) {
      photonIdVals[iid].push_back( (*photonIdHandles[iid])[photonref] );
    }    

    for (unsigned int iid=0; iid<photonPFClusterIsoHandles.size(); ++iid) {
      photonPFClusterIsoVals[iid].push_back( (*photonPFClusterIsoHandles[iid])[photonref] );
    }    
    
    //link photon core
    const reco::PhotonCoreRef &photonCore = photon.photonCore();
    linkCore(photonCore, *photonCores, photonCoreMap);
    
    bool slimRelink = slimRelinkPhotonSel_(photon);
    //no supercluster relinking unless slimRelink selection is satisfied
    if (!slimRelink) continue;
    
    bool relink = relinkPhotonSel_(photon);
    
    //link supercluster
    const reco::SuperClusterRef &superCluster = photon.superCluster();
    linkSuperCluster(superCluster, superClusterMap, *superClusters, relink, superClusterFullRelinkMap);
    
    //conversions only for full relinking
    if (!relink) continue;
    
    const reco::ConversionRefVector &convrefs = photon.conversions();
    linkConversions(convrefs, *conversions, conversionMap);

    //explicitly references conversions
    const reco::ConversionRefVector &singleconvrefs = photon.conversionsOneLeg();
    linkConversions(singleconvrefs, *singleConversions, singleConversionMap);
  }

  //loop over oot photons and fill maps
  //special note1: since not PFCand --> no PF isolation, IDs (but we do have PFClusterIso!)
  //special note2: conversion sequence not run over bcs from oot phos, so skip relinking of oot phos
  //special note3: clusters and superclusters in own collections!
  for (unsigned int iootpho=0; iootpho<ootPhotonHandle->size(); ++iootpho) {
    const reco::Photon &ootPhoton = (*ootPhotonHandle)[iootpho];
    
    bool keep = keepOOTPhotonSel_(ootPhoton);
    if (!keep) continue;
    
    reco::PhotonRef ootPhotonref(ootPhotonHandle,iootpho);
    
    ootPhotons->push_back(ootPhoton);
        
    //fill photon pfclusteriso valuemap vectors
    for (unsigned int iid=0; iid<ootPhotonPFClusterIsoHandles.size(); ++iid) {
      ootPhotonPFClusterIsoVals[iid].push_back( (*ootPhotonPFClusterIsoHandles[iid])[ootPhotonref] );
    }    

    //link photon core
    const reco::PhotonCoreRef &ootPhotonCore = ootPhoton.photonCore();
    linkCore(ootPhotonCore, *ootPhotonCores, ootPhotonCoreMap);

    bool slimRelink = slimRelinkOOTPhotonSel_(ootPhoton);
    //no supercluster relinking unless slimRelink selection is satisfied
    if (!slimRelink) continue;
    
    bool relink = relinkOOTPhotonSel_(ootPhoton);
    
    const reco::SuperClusterRef &ootSuperCluster = ootPhoton.superCluster();
    linkSuperCluster(ootSuperCluster, ootSuperClusterMap, *ootSuperClusters, relink, ootSuperClusterFullRelinkMap);
  }
  
  //loop over electrons and fill maps
  for (unsigned int iele = 0; iele<gsfElectronHandle->size(); ++iele) {
    const reco::GsfElectron &gsfElectron = (*gsfElectronHandle)[iele];
    
    bool keep = keepGsfElectronSel_(gsfElectron);    
    if (!keep) continue;
    
    reco::GsfElectronRef gsfElectronref(gsfElectronHandle,iele);
    
    gsfElectrons->push_back(gsfElectron);
    pfCandIsoPairVecEle.push_back((*gsfElectronPfCandMapHandle)[gsfElectronref]);
    
    //fill electron id valuemap vectors
    for (unsigned int iid=0; iid<gsfElectronIdHandles.size(); ++iid) {
      gsfElectronIdVals[iid].push_back( (*gsfElectronIdHandles[iid])[gsfElectronref] );
    }    

    for (unsigned int iid=0; iid<gsfElectronPFClusterIsoHandles.size(); ++iid) {
      gsfElectronPFClusterIsoVals[iid].push_back( (*gsfElectronPFClusterIsoHandles[iid])[gsfElectronref] );
    }    

    const reco::GsfElectronCoreRef &gsfElectronCore = gsfElectron.core();
    linkCore(gsfElectronCore, *gsfElectronCores, gsfElectronCoreMap);

    bool slimRelink = slimRelinkGsfElectronSel_(gsfElectron);
    //no supercluster relinking unless slimRelink selection is satisfied
    if (!slimRelink) continue;
    
    bool relink = relinkGsfElectronSel_(gsfElectron);
    
    const reco::SuperClusterRef &superCluster = gsfElectron.superCluster();
    linkSuperCluster(superCluster, superClusterMap, *superClusters, relink, superClusterFullRelinkMap);    

    //conversions only for full relinking
    if (!relink) continue;
    
    const reco::ConversionRefVector &convrefs = gsfElectron.core()->conversions();
    linkConversions(convrefs, *conversions, conversionMap);
    
    //explicitly references conversions
    const reco::ConversionRefVector &singleconvrefs = gsfElectron.core()->conversionsOneLeg();
    linkConversions(singleconvrefs, *singleConversions, singleConversionMap);

    //conversions matched by trackrefs
    linkConversionsByTrackRef(conversionHandle, gsfElectron, *conversions, conversionMap);
    
    //single leg conversions matched by trackrefs
    linkConversionsByTrackRef(singleConversionHandle, gsfElectron, *singleConversions, singleConversionMap);
  }

  //loop over output SuperClusters and fill maps
  for (unsigned int isc = 0; isc<superClusters->size(); ++isc) {
    reco::SuperCluster &superCluster = (*superClusters)[isc];
    
    //link seed cluster no matter what
    const reco::CaloClusterPtr &seedCluster = superCluster.seed();
    linkCaloCluster(seedCluster, *ebeeClusters, ebeeClusterMap);
        
    //only proceed if superCluster is marked for full relinking
    bool fullrelink = superClusterFullRelinkMap.count(isc);
    if (!fullrelink) {
      //zero detid vector which is anyways not useful without stored rechits
      superCluster.clearHitsAndFractions();
      continue; 
    }
    
    // link calo clusters
    linkCaloClusters(superCluster, *ebeeClusters, ebeeClusterMap, rechitMap, barrelHitHandle, endcapHitHandle, caloTopology, *esClusters, esClusterMap);
    
    //conversions matched geometrically
    linkConversionsByTrackRef(conversionHandle, superCluster, *conversions, conversionMap);
    
    //single leg conversions matched by trackrefs
    linkConversionsByTrackRef(singleConversionHandle, superCluster, *singleConversions, singleConversionMap);    
  }
  
  //loop over output OOTSuperClusters and fill maps
  for (unsigned int isc = 0; isc<ootSuperClusters->size(); ++isc) {
    reco::SuperCluster &ootSuperCluster = (*ootSuperClusters)[isc];
    
    //link seed cluster no matter what
    const reco::CaloClusterPtr &ootSeedCluster = ootSuperCluster.seed();
    linkCaloCluster(ootSeedCluster, *ootEbeeClusters, ootEbeeClusterMap);

    //only proceed if ootSuperCluster is marked for full relinking
    bool fullrelink = ootSuperClusterFullRelinkMap.count(isc);
    if (!fullrelink) {
      //zero detid vector which is anyways not useful without stored rechits
      ootSuperCluster.clearHitsAndFractions();
      continue; 
    }

    // link calo clusters
    linkCaloClusters(ootSuperCluster, *ootEbeeClusters, ootEbeeClusterMap, rechitMap, barrelHitHandle, endcapHitHandle, caloTopology, *ootEsClusters, ootEsClusterMap);
  }

  //now finalize and add to the event collections in "reverse" order
  
  //rechits (fill output collections of rechits to be stored)
  for (const EcalRecHit &rechit : *barrelHitHandle) {
    if (rechitMap.count(rechit.detid())) {
      ebRecHits->push_back(rechit);
    }
  }
  
  for (const EcalRecHit &rechit : *endcapHitHandle) {
    if (rechitMap.count(rechit.detid())) {
      eeRecHits->push_back(rechit);
    }
  }
 
  theEvent.put(std::move(ebRecHits),outEBRecHits_);
  theEvent.put(std::move(eeRecHits),outEERecHits_);

  if (doPreshowerEcalHits_) { 
    for (const EcalRecHit &rechit : *preshowerHitHandle) {
      if (rechitMap.count(rechit.detid())) {
	esRecHits->push_back(rechit);
      }
    }
    theEvent.put(std::move(esRecHits),outESRecHits_);  
  }
  
  //CaloClusters
  //put calocluster output collections in event and get orphan handles to create ptrs
  const edm::OrphanHandle<reco::CaloClusterCollection> &outEBEEClusterHandle = theEvent.put(std::move(ebeeClusters),outEBEEClusters_);
  const edm::OrphanHandle<reco::CaloClusterCollection> &outESClusterHandle = theEvent.put(std::move(esClusters),outESClusters_);;  

  //Loop over SuperClusters and relink GEDPhoton + GSFElectron CaloClusters
  for (reco::SuperCluster &superCluster : *superClusters) {
    relinkCaloClusters(superCluster, ebeeClusterMap, esClusterMap, outEBEEClusterHandle, outESClusterHandle);
  }

  //OOTCaloClusters
  //put ootcalocluster output collections in event and get orphan handles to create ptrs
  const edm::OrphanHandle<reco::CaloClusterCollection> &outOOTEBEEClusterHandle = theEvent.put(std::move(ootEbeeClusters),outOOTEBEEClusters_);
  const edm::OrphanHandle<reco::CaloClusterCollection> &outOOTESClusterHandle = theEvent.put(std::move(ootEsClusters),outOOTESClusters_);;  
  
  //Loop over OOTSuperClusters and relink OOTPhoton CaloClusters
  for (reco::SuperCluster &ootSuperCluster : *ootSuperClusters) {
    relinkCaloClusters(ootSuperCluster, ootEbeeClusterMap, ootEsClusterMap, outOOTEBEEClusterHandle, outOOTESClusterHandle);
  }

  //put superclusters and conversions in the event
  const edm::OrphanHandle<reco::SuperClusterCollection> &outSuperClusterHandle = theEvent.put(std::move(superClusters),outSuperClusters_);
  const edm::OrphanHandle<reco::ConversionCollection> &outConversionHandle = theEvent.put(std::move(conversions),outConversions_);
  const edm::OrphanHandle<reco::ConversionCollection> &outSingleConversionHandle = theEvent.put(std::move(singleConversions),outSingleConversions_);

  //Loop over PhotonCores and relink GEDPhoton SuperClusters (and conversions)
  for (reco::PhotonCore &photonCore : *photonCores) {
    // superclusters
    relinkSuperCluster(photonCore, superClusterMap, outSuperClusterHandle);

    //conversions
    const reco::ConversionRefVector &convrefs = photonCore.conversions();
    relinkConversions(photonCore, convrefs, conversionMap, outConversionHandle);

    //single leg conversions
    const reco::ConversionRefVector &singleconvrefs = photonCore.conversionsOneLeg();
    relinkConversions(photonCore, singleconvrefs, singleConversionMap, outSingleConversionHandle);
  }

  //Relink GSFElectron SuperClusters
  for (reco::GsfElectronCore &gsfElectronCore : *gsfElectronCores) {
    relinkSuperCluster(gsfElectronCore, superClusterMap, outSuperClusterHandle);
  }
  
  //put ootsuperclusters in the event
  const edm::OrphanHandle<reco::SuperClusterCollection> &outOOTSuperClusterHandle = theEvent.put(std::move(ootSuperClusters),outOOTSuperClusters_);
  
  //Relink OOTPhoton SuperClusters
  for (reco::PhotonCore &ootPhotonCore : *ootPhotonCores) {
    relinkSuperCluster(ootPhotonCore, ootSuperClusterMap, outOOTSuperClusterHandle);
  }

  //put photoncores and gsfelectroncores into the event
  const edm::OrphanHandle<reco::PhotonCoreCollection> &outPhotonCoreHandle = theEvent.put(std::move(photonCores),outPhotonCores_);
  const edm::OrphanHandle<reco::PhotonCoreCollection> &outOOTPhotonCoreHandle = theEvent.put(std::move(ootPhotonCores),outOOTPhotonCores_);
  const edm::OrphanHandle<reco::GsfElectronCoreCollection> &outgsfElectronCoreHandle = theEvent.put(std::move(gsfElectronCores),outGsfElectronCores_);
  
  //loop over photons, oot photons, and electrons and relink the cores
  for (reco::Photon &photon : *photons) {
    relinkPhotonCore(photon, photonCoreMap, outPhotonCoreHandle);
  }

  for (reco::Photon &ootPhoton : *ootPhotons) {
    relinkPhotonCore(ootPhoton, ootPhotonCoreMap, outOOTPhotonCoreHandle);
  }

  for (reco::GsfElectron &gsfElectron : *gsfElectrons) {
    relinkGsfElectronCore(gsfElectron, gsfElectronCoreMap, outgsfElectronCoreHandle);
  }
  
  //(finally) store the output photon and electron collections
  const edm::OrphanHandle<reco::PhotonCollection> &outPhotonHandle = theEvent.put(std::move(photons),outPhotons_);  
  const edm::OrphanHandle<reco::PhotonCollection> &outOOTPhotonHandle = theEvent.put(std::move(ootPhotons),outOOTPhotons_);  
  const edm::OrphanHandle<reco::GsfElectronCollection> &outGsfElectronHandle = theEvent.put(std::move(gsfElectrons),outGsfElectrons_);
  
  //still need to output relinked valuemaps
  
  //photon pfcand isolation valuemap
  edm::ValueMap<std::vector<reco::PFCandidateRef>>::Filler fillerPhotons(*photonPfCandMap);
  fillerPhotons.insert(outPhotonHandle,pfCandIsoPairVecPho.begin(),pfCandIsoPairVecPho.end());
  fillerPhotons.fill();   
  
  //electron pfcand isolation valuemap
  edm::ValueMap<std::vector<reco::PFCandidateRef>>::Filler fillerGsfElectrons(*gsfElectronPfCandMap);
  fillerGsfElectrons.insert(outGsfElectronHandle,pfCandIsoPairVecEle.begin(),pfCandIsoPairVecEle.end());
  fillerGsfElectrons.fill();
  
  theEvent.put(std::move(photonPfCandMap),outPhotonPfCandMap_);
  theEvent.put(std::move(gsfElectronPfCandMap),outGsfElectronPfCandMap_);
  
  auto fillMap = [](auto refH, auto& vec, edm::Event& ev, const std::string& cAl = ""){
    typedef  edm::ValueMap<typename std::decay<decltype(vec)>::type::value_type> MapType;
    auto oMap = std::make_unique<MapType>();
    {
      typename MapType::Filler filler(*oMap);
      filler.insert(refH, vec.begin(), vec.end());
      filler.fill();
    }
    ev.put(std::move(oMap), cAl);
  };
  int index = 0;

  //photon id value maps
  for (auto const& vals : photonIdVals){ 
    fillMap(outPhotonHandle, vals, theEvent, outPhotonIds_[index]);
    index++;
  }
  index = 0;

  //electron id value maps
  for (auto const& vals : gsfElectronIdVals){ 
    fillMap(outGsfElectronHandle, vals, theEvent, outGsfElectronIds_[index]);
    index++;
  }
  index = 0;
  
  // photon iso value maps
  for (auto const& vals : photonPFClusterIsoVals){
    fillMap(outPhotonHandle, vals, theEvent, outPhotonPFClusterIsos_[index]);
    index++;
  }
  index = 0;

  //oot photon iso value maps
  for (auto const& vals : ootPhotonPFClusterIsoVals){
    fillMap(outOOTPhotonHandle, vals, theEvent, outOOTPhotonPFClusterIsos_[index]);
    index++;
  }
  index = 0;

  //electron iso value maps
  for (auto const& vals : gsfElectronPFClusterIsoVals){
    fillMap(outGsfElectronHandle, vals, theEvent, outGsfElectronPFClusterIsos_[index]);
    index++;
  }
  index = 0;
}

template <typename T, typename U>
void ReducedEGProducer::linkCore(const T& core, U& cores, std::map<T, unsigned int>& coreMap)
{
  if (!coreMap.count(core)) {
    cores.push_back(*core);
    coreMap[core] = cores.size() - 1;
  }
}
    
void ReducedEGProducer::linkSuperCluster(const reco::SuperClusterRef& superCluster, 
					 std::map<reco::SuperClusterRef, unsigned int>& superClusterMap,
					 reco::SuperClusterCollection& superClusters,
					 const bool relink,
					 std::unordered_set<unsigned int>& superClusterFullRelinkMap)
{
  const auto &mappedsc = superClusterMap.find(superCluster);
  //get index in output collection in order to keep track whether superCluster
  //will be subject to full relinking
  unsigned int mappedscidx = 0;
  if (mappedsc==superClusterMap.end()) {
    superClusters.push_back(*superCluster);
    mappedscidx = superClusters.size() - 1;
    superClusterMap[superCluster] = mappedscidx;
  }
  else {
    mappedscidx = mappedsc->second;
  }

  //additionally mark supercluster for full relinking
  if (relink) superClusterFullRelinkMap.insert(mappedscidx);
}

void ReducedEGProducer::linkConversions(const reco::ConversionRefVector& convrefs,
					reco::ConversionCollection& conversions, 
					std::map<reco::ConversionRef, unsigned int>& conversionMap)
{
  for (const auto& convref : convrefs) {
    linkConversion(convref, conversions, conversionMap);
  }
}

void ReducedEGProducer::linkConversionsByTrackRef(const edm::Handle<reco::ConversionCollection>& conversionHandle,
						  const reco::GsfElectron& gsfElectron,
						  reco::ConversionCollection& conversions, 
						  std::map<reco::ConversionRef, unsigned int>& conversionMap)
{
  for (unsigned int iconv = 0; iconv<conversionHandle->size(); ++iconv) {
    const reco::Conversion &conversion = (*conversionHandle)[iconv];
    reco::ConversionRef convref(conversionHandle,iconv);
    
    bool matched = ConversionTools::matchesConversion(gsfElectron,conversion,true,true);
    if (!matched) continue;
      
    linkConversion(convref, conversions, conversionMap);
  }
}

void ReducedEGProducer::linkConversionsByTrackRef(const edm::Handle<reco::ConversionCollection>& conversionHandle,
						  const reco::SuperCluster& superCluster,
						  reco::ConversionCollection& conversions, 
						  std::map<reco::ConversionRef, unsigned int>& conversionMap)
{
  for (unsigned int iconv = 0; iconv<conversionHandle->size(); ++iconv) {
    const reco::Conversion &conversion = (*conversionHandle)[iconv];
    reco::ConversionRef convref(conversionHandle,iconv);
    
    bool matched = ConversionTools::matchesConversion(superCluster,conversion,0.2);
    if (!matched) continue;
 
    linkConversion(convref, conversions, conversionMap);
  }
}

void ReducedEGProducer::linkConversion(const reco::ConversionRef& convref,
				       reco::ConversionCollection& conversions, 
				       std::map<reco::ConversionRef, unsigned int>& conversionMap)
{
  if (!conversionMap.count(convref)) {
    conversions.push_back(*convref);
    conversionMap[convref] = conversions.size() - 1;
  }
}

void ReducedEGProducer::linkCaloCluster(const reco::CaloClusterPtr& caloCluster, 
					reco::CaloClusterCollection& caloClusters,
					std::map<reco::CaloClusterPtr, unsigned int>& caloClusterMap)
{
  if (!caloClusterMap.count(caloCluster)) {
    caloClusters.push_back(*caloCluster);
    caloClusterMap[caloCluster] = caloClusters.size() - 1;
  }
}

void ReducedEGProducer::linkCaloClusters(const reco::SuperCluster& superCluster, 
					 reco::CaloClusterCollection& ebeeClusters,
					 std::map<reco::CaloClusterPtr, unsigned int>& ebeeClusterMap,
					 std::unordered_set<DetId>& rechitMap,
					 const edm::Handle<EcalRecHitCollection>& barrelHitHandle,
					 const edm::Handle<EcalRecHitCollection>& endcapHitHandle,
					 const CaloTopology *caloTopology,
					 reco::CaloClusterCollection& esClusters,
					 std::map<reco::CaloClusterPtr, unsigned int>& esClusterMap)
{
  for (const auto& cluster : superCluster.clusters()) {
    linkCaloCluster(cluster, ebeeClusters, ebeeClusterMap);

    for (const auto& hitfrac : cluster->hitsAndFractions()) {
      rechitMap.insert(hitfrac.first);
    }
    //make sure to also take all hits in the 5x5 around the max energy xtal
    bool barrel = cluster->hitsAndFractions().front().first.subdetId()==EcalBarrel;
    const EcalRecHitCollection *rhcol = barrel ? barrelHitHandle.product() : endcapHitHandle.product();
    DetId seed = EcalClusterTools::getMaximum(*cluster, rhcol).first;
    
    std::vector<DetId> dets5x5 = caloTopology->getSubdetectorTopology(DetId::Ecal, barrel ? EcalBarrel : EcalEndcap)->getWindow(seed,5,5);
    for (const auto& detid : dets5x5) {
      rechitMap.insert(detid);
    }
  }
  for (const auto& cluster : superCluster.preshowerClusters()) {
    linkCaloCluster(cluster, esClusters, esClusterMap);

    for (const auto& hitfrac : cluster->hitsAndFractions()) {
      rechitMap.insert(hitfrac.first);
    }      
  }
}


void ReducedEGProducer::relinkCaloClusters(reco::SuperCluster& superCluster, 
					   const std::map<reco::CaloClusterPtr, unsigned int>& ebeeClusterMap, 
					   const std::map<reco::CaloClusterPtr, unsigned int>& esClusterMap, 
					   const edm::OrphanHandle<reco::CaloClusterCollection>& outEBEEClusterHandle, 
					   const edm::OrphanHandle<reco::CaloClusterCollection>& outESClusterHandle)
{
  //remap seed cluster
  const auto &seedmapped = ebeeClusterMap.find(superCluster.seed());
  if (seedmapped != ebeeClusterMap.end()) {
    //make new ptr
    reco::CaloClusterPtr clusptr(outEBEEClusterHandle,seedmapped->second);
    superCluster.setSeed(clusptr);
  }

 //remap all clusters
  reco::CaloClusterPtrVector clusters;
  for (const auto &cluster : superCluster.clusters()) {
    const auto &clustermapped = ebeeClusterMap.find(cluster);
    if (clustermapped != ebeeClusterMap.end()) {
      //make new ptr
      reco::CaloClusterPtr clusptr(outEBEEClusterHandle,clustermapped->second);
      clusters.push_back(clusptr);
    }
    else {
      //can only relink if all clusters are being relinked, so if one is missing, then skip the relinking completely
      clusters.clear();
      break;
    }
  }
  if (clusters.size()) {
    superCluster.setClusters(clusters);
  }
  
  //remap preshower clusters
  reco::CaloClusterPtrVector esclusters;
  for (const auto& cluster : superCluster.preshowerClusters()) {
    const auto &clustermapped = esClusterMap.find(cluster);
    if (clustermapped != esClusterMap.end()) {
      //make new ptr
      reco::CaloClusterPtr clusptr(outESClusterHandle,clustermapped->second);
      esclusters.push_back(clusptr);
    }
    else {
      //can only relink if all clusters are being relinked, so if one is missing, then skip the relinking completely
      esclusters.clear();
      break;
    }
  }
  if (esclusters.size()) {
    superCluster.setPreshowerClusters(esclusters);
  }
}

template <typename T>
void ReducedEGProducer::relinkSuperCluster(T& core, 
					   const std::map<reco::SuperClusterRef, unsigned int>& superClusterMap, 
					   const edm::OrphanHandle<reco::SuperClusterCollection>& outSuperClusterHandle) 
{
  const auto &scmapped = superClusterMap.find(core.superCluster());
  if (scmapped != superClusterMap.end()) {
    //make new ref
    reco::SuperClusterRef scref(outSuperClusterHandle,scmapped->second);
    core.setSuperCluster(scref);
  }
}   

void ReducedEGProducer::relinkConversions(reco::PhotonCore& photonCore, 
					  const reco::ConversionRefVector& convrefs,
					  const std::map<reco::ConversionRef, unsigned int>& conversionMap,
					  const edm::OrphanHandle<reco::ConversionCollection> &outConversionHandle)
{
  reco::ConversionRefVector outconvrefs;
  for (const auto& convref : convrefs) {
    const auto &convmapped = conversionMap.find(convref);
    if (convmapped != conversionMap.end()) {
      //make new ref
        reco::ConversionRef outref(outConversionHandle,convmapped->second);
    }
    else {
      //can only relink if all conversions are being relinked, so if one is missing, then skip the relinking completely
      outconvrefs.clear();
      break;
    }
  }
  if (outconvrefs.size()) {
    photonCore.setConversions(outconvrefs);
  }
}

void ReducedEGProducer::relinkPhotonCore(reco::Photon& photon, 
					 const std::map<reco::PhotonCoreRef, unsigned int>& photonCoreMap, 
					 const edm::OrphanHandle<reco::PhotonCoreCollection>& outPhotonCoreHandle) 
{
  const auto &coremapped = photonCoreMap.find(photon.photonCore());
  if (coremapped != photonCoreMap.end()) {
    //make new ref
    reco::PhotonCoreRef coreref(outPhotonCoreHandle,coremapped->second);
    photon.setPhotonCore(coreref);
  }
}

void ReducedEGProducer::relinkGsfElectronCore(reco::GsfElectron& gsfElectron,
					      const std::map<reco::GsfElectronCoreRef, unsigned int>& gsfElectronCoreMap, 
					      const edm::OrphanHandle<reco::GsfElectronCoreCollection>& outgsfElectronCoreHandle) 
{
  const auto &coremapped = gsfElectronCoreMap.find(gsfElectron.core());
  if (coremapped != gsfElectronCoreMap.end()) {
    //make new ref
    reco::GsfElectronCoreRef coreref(outgsfElectronCoreHandle,coremapped->second);
    gsfElectron.setCore(coreref);
  }
}
