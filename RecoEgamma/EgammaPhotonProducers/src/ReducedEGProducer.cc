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
  gsfElectronT_(consumes<reco::GsfElectronCollection>(config.getParameter<edm::InputTag>("gsfElectrons"))),
  conversionT_(consumes<reco::ConversionCollection>(config.getParameter<edm::InputTag>("conversions"))),
  singleConversionT_(consumes<reco::ConversionCollection>(config.getParameter<edm::InputTag>("singleConversions"))),
  barrelEcalHits_(consumes<EcalRecHitCollection>(config.getParameter<edm::InputTag>("barrelEcalHits"))),
  endcapEcalHits_(consumes<EcalRecHitCollection>(config.getParameter<edm::InputTag>("endcapEcalHits"))),
  preshowerEcalHits_(consumes<EcalRecHitCollection>(config.getParameter<edm::InputTag>("preshowerEcalHits"))),
  photonPfCandMapT_(consumes<edm::ValueMap<std::vector<reco::PFCandidateRef> > >(config.getParameter<edm::InputTag>("photonsPFValMap"))),  
  gsfElectronPfCandMapT_(consumes<edm::ValueMap<std::vector<reco::PFCandidateRef> > >(config.getParameter<edm::InputTag>("gsfElectronsPFValMap"))),
  //output collections    
  outPhotons_("reducedGedPhotons"),
  outPhotonCores_("reducedGedPhotonCores"),
  outGsfElectrons_("reducedGedGsfElectrons"),
  outGsfElectronCores_("reducedGedGsfElectronCores"),
  outConversions_("reducedConversions"),
  outSingleConversions_("reducedSingleLegConversions"),
  outSuperClusters_("reducedSuperClusters"),
  outEBEEClusters_("reducedEBEEClusters"),
  outESClusters_("reducedESClusters"),
  outEBRecHits_("reducedEBRecHits"),
  outEERecHits_("reducedEERecHits"),
  outESRecHits_("reducedESRecHits"),
  outPhotonPfCandMap_("reducedPhotonPfCandMap"),
  outGsfElectronPfCandMap_("reducedGsfElectronPfCandMap"),
  outPhotonIds_(config.getParameter<std::vector<std::string> >("photonIDOutput")),
  outGsfElectronIds_(config.getParameter<std::vector<std::string> >("gsfElectronIDOutput")),
  outPhotonPFClusterIsos_(config.getParameter<std::vector<std::string> >("photonPFClusterIsoOutput")),
  outGsfElectronPFClusterIsos_(config.getParameter<std::vector<std::string> >("gsfElectronPFClusterIsoOutput")),
  keepPhotonSel_(config.getParameter<std::string>("keepPhotons")),
  slimRelinkPhotonSel_(config.getParameter<std::string>("slimRelinkPhotons")),
  relinkPhotonSel_(config.getParameter<std::string>("relinkPhotons")),
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

  const std::vector<edm::InputTag>& gsfelectronpfclusterisoinputs = 
    config.getParameter<std::vector<edm::InputTag> >("gsfElectronPFClusterIsoSources");
  for (const edm::InputTag &tag : gsfelectronpfclusterisoinputs) {
    gsfElectronPFClusterIsoTs_.emplace_back(consumes<edm::ValueMap<float> >(tag));
  }  
  
  produces< reco::PhotonCollection >(outPhotons_);
  produces< reco::PhotonCoreCollection >(outPhotonCores_);
  produces< reco::GsfElectronCollection >(outGsfElectrons_);
  produces< reco::GsfElectronCoreCollection >(outGsfElectronCores_);
  produces< reco::ConversionCollection >(outConversions_);
  produces< reco::ConversionCollection >(outSingleConversions_);
  produces< reco::SuperClusterCollection >(outSuperClusters_);  
  produces< reco::CaloClusterCollection >(outEBEEClusters_);
  produces< reco::CaloClusterCollection >(outESClusters_);
  produces< EcalRecHitCollection >(outEBRecHits_);
  produces< EcalRecHitCollection >(outEERecHits_);
  produces< EcalRecHitCollection >(outESRecHits_);    
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
  theEvent.getByToken(preshowerEcalHits_, preshowerHitHandle);
  
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
  
  edm::ESHandle<CaloTopology> theCaloTopology;
  theEventSetup.get<CaloTopologyRecord>().get(theCaloTopology);  
  const CaloTopology *caloTopology = & (*theCaloTopology);  
  
  //initialize output collections
  std::auto_ptr<reco::PhotonCollection> photons(new reco::PhotonCollection);
  std::auto_ptr<reco::PhotonCoreCollection> photonCores(new reco::PhotonCoreCollection);
  std::auto_ptr<reco::GsfElectronCollection> gsfElectrons(new reco::GsfElectronCollection);
  std::auto_ptr<reco::GsfElectronCoreCollection> gsfElectronCores(new reco::GsfElectronCoreCollection);
  std::auto_ptr<reco::ConversionCollection> conversions(new reco::ConversionCollection);
  std::auto_ptr<reco::ConversionCollection> singleConversions(new reco::ConversionCollection);
  std::auto_ptr<reco::SuperClusterCollection> superClusters(new reco::SuperClusterCollection);
  std::auto_ptr<reco::CaloClusterCollection> ebeeClusters(new reco::CaloClusterCollection);
  std::auto_ptr<reco::CaloClusterCollection> esClusters(new reco::CaloClusterCollection);
  std::auto_ptr<EcalRecHitCollection> ebRecHits(new EcalRecHitCollection);
  std::auto_ptr<EcalRecHitCollection> eeRecHits(new EcalRecHitCollection);
  std::auto_ptr<EcalRecHitCollection> esRecHits(new EcalRecHitCollection);
  std::auto_ptr<edm::ValueMap<std::vector<reco::PFCandidateRef> > > photonPfCandMap(new edm::ValueMap<std::vector<reco::PFCandidateRef> >);
  std::auto_ptr<edm::ValueMap<std::vector<reco::PFCandidateRef> > > gsfElectronPfCandMap(new edm::ValueMap<std::vector<reco::PFCandidateRef> >);
  
  std::vector<std::auto_ptr<edm::ValueMap<bool> > > photonIds;
  for (unsigned int iid=0; iid<photonIdHandles.size(); ++iid) {
    photonIds.emplace_back(new edm::ValueMap<bool>);
  }
    
  std::vector<std::auto_ptr<edm::ValueMap<float> > > gsfElectronIds;
  for (unsigned int iid=0; iid<gsfElectronIdHandles.size(); ++iid) {
    gsfElectronIds.emplace_back(new edm::ValueMap<float>);
  }

  std::vector<std::auto_ptr<edm::ValueMap<float> > > photonPFClusterIsos;
  for (unsigned int iid=0; iid<photonPFClusterIsoHandles.size(); ++iid) {
    photonPFClusterIsos.emplace_back(new edm::ValueMap<float>);
  }

  std::vector<std::auto_ptr<edm::ValueMap<float> > > gsfElectronPFClusterIsos;
  for (unsigned int iid=0; iid<gsfElectronPFClusterIsoHandles.size(); ++iid) {
    gsfElectronPFClusterIsos.emplace_back(new edm::ValueMap<float>);
  }
 
  //maps to collection indices of output objects
  std::map<reco::PhotonCoreRef, unsigned int> photonCoreMap;
  std::map<reco::GsfElectronCoreRef, unsigned int> gsfElectronCoreMap;
  std::map<reco::ConversionRef, unsigned int> conversionMap;
  std::map<reco::ConversionRef, unsigned int> singleConversionMap;
  std::map<reco::SuperClusterRef, unsigned int> superClusterMap;
  std::map<reco::CaloClusterPtr, unsigned int> ebeeClusterMap;
  std::map<reco::CaloClusterPtr, unsigned int> esClusterMap;
  std::unordered_set<DetId> rechitMap;
  
  std::unordered_set<unsigned int> superClusterFullRelinkMap;
  
  //vectors for pfcandidate valuemaps
  std::vector<std::vector<reco::PFCandidateRef> > pfCandIsoPairVecPho;  
  std::vector<std::vector<reco::PFCandidateRef> > pfCandIsoPairVecEle;
  
  //vectors for id valuemaps
  std::vector<std::vector<bool> > photonIdVals(photonIds.size());
  std::vector<std::vector<float> > gsfElectronIdVals(gsfElectronIds.size());
  std::vector<std::vector<float> > photonPFClusterIsoVals(photonPFClusterIsos.size());
  std::vector<std::vector<float> > gsfElectronPFClusterIsoVals(gsfElectronPFClusterIsos.size());
  
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
    for (unsigned int iid=0; iid<photonIds.size(); ++iid) {
      photonIdVals[iid].push_back( (*photonIdHandles[iid])[photonref] );
    }    

    for (unsigned int iid=0; iid<photonPFClusterIsos.size(); ++iid) {
      photonPFClusterIsoVals[iid].push_back( (*photonPFClusterIsoHandles[iid])[photonref] );
    }    
    
    const reco::PhotonCoreRef &photonCore = photon.photonCore();
    if (!photonCoreMap.count(photonCore)) {
      photonCores->push_back(*photonCore);
      photonCoreMap[photonCore] = photonCores->size() - 1;
    }
    
    bool slimRelink = slimRelinkPhotonSel_(photon);
    //no supercluster relinking unless slimRelink selection is satisfied
    if (!slimRelink) continue;
    
    bool relink = relinkPhotonSel_(photon);
    
    const reco::SuperClusterRef &superCluster = photon.superCluster();
    const auto &mappedsc = superClusterMap.find(superCluster);
    //get index in output collection in order to keep track whether superCluster
    //will be subject to full relinking
    unsigned int mappedscidx = 0;
    if (mappedsc==superClusterMap.end()) {
      superClusters->push_back(*superCluster);
      mappedscidx = superClusters->size() - 1;
      superClusterMap[superCluster] = mappedscidx;
    }
    else {
      mappedscidx = mappedsc->second;
    }
    
    //additionally mark supercluster for full relinking
    if (relink) superClusterFullRelinkMap.insert(mappedscidx);
    
    //conversions only for full relinking
    if (!relink) continue;
    
    const reco::ConversionRefVector &convrefs = photon.conversions();
    for (const reco::ConversionRef &convref : convrefs) {
      if (!conversionMap.count(convref)) {
        conversions->push_back(*convref);
        conversionMap[convref] = conversions->size() - 1;
      }
    }
    
    //explicitly references conversions
    const reco::ConversionRefVector &singleconvrefs = photon.conversionsOneLeg();
    for (const reco::ConversionRef &convref : singleconvrefs) {
      if (!singleConversionMap.count(convref)) {
        singleConversions->push_back(*convref);
        singleConversionMap[convref] = singleConversions->size() - 1;
      }
    }    
    
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
    for (unsigned int iid=0; iid<gsfElectronIds.size(); ++iid) {
      gsfElectronIdVals[iid].push_back( (*gsfElectronIdHandles[iid])[gsfElectronref] );
    }    

    for (unsigned int iid=0; iid<gsfElectronPFClusterIsos.size(); ++iid) {
      gsfElectronPFClusterIsoVals[iid].push_back( (*gsfElectronPFClusterIsoHandles[iid])[gsfElectronref] );
    }    

    const reco::GsfElectronCoreRef &gsfElectronCore = gsfElectron.core();
    if (!gsfElectronCoreMap.count(gsfElectronCore)) {
      gsfElectronCores->push_back(*gsfElectronCore);
      gsfElectronCoreMap[gsfElectronCore] = gsfElectronCores->size() - 1;
    }    
    
    bool slimRelink = slimRelinkGsfElectronSel_(gsfElectron);
    //no supercluster relinking unless slimRelink selection is satisfied
    if (!slimRelink) continue;
    
    bool relink = relinkGsfElectronSel_(gsfElectron);
    
    const reco::SuperClusterRef &superCluster = gsfElectron.superCluster();
    const auto &mappedsc = superClusterMap.find(superCluster);
    //get index in output collection in order to keep track whether superCluster
    //will be subject to full relinking
    unsigned int mappedscidx = 0;
    if (mappedsc==superClusterMap.end()) {
      superClusters->push_back(*superCluster);
      mappedscidx = superClusters->size() - 1;
      superClusterMap[superCluster] = mappedscidx;
    }
    else {
      mappedscidx = mappedsc->second;
    }
    
    //additionally mark supercluster for full relinking
    if (relink) superClusterFullRelinkMap.insert(mappedscidx);
    
    //conversions only for full relinking
    if (!relink) continue;
    
    const reco::ConversionRefVector &convrefs = gsfElectron.core()->conversions();
    for (const reco::ConversionRef &convref : convrefs) {
      if (!conversionMap.count(convref)) {
        conversions->push_back(*convref);
        conversionMap[convref] = conversions->size() - 1;
      }
    }
    
    //explicitly references conversions
    const reco::ConversionRefVector &singleconvrefs = gsfElectron.core()->conversionsOneLeg();
    for (const reco::ConversionRef &convref : singleconvrefs) {
      if (!singleConversionMap.count(convref)) {
        singleConversions->push_back(*convref);
        singleConversionMap[convref] = singleConversions->size() - 1;
      }
    }     
    
    //conversions matched by trackrefs
    for (unsigned int iconv = 0; iconv<conversionHandle->size(); ++iconv) {
      const reco::Conversion &conversion = (*conversionHandle)[iconv];
      reco::ConversionRef convref(conversionHandle,iconv);
      
      bool matched = ConversionTools::matchesConversion(gsfElectron,conversion,true,true);
      if (!matched) continue;
      
      if (!conversionMap.count(convref)) {
        conversions->push_back(conversion);
        conversionMap[convref] = conversions->size() - 1;
      }
      
    }
    
    //single leg conversions matched by trackrefs
    for (unsigned int iconv = 0; iconv<singleConversionHandle->size(); ++iconv) {
      const reco::Conversion &conversion = (*singleConversionHandle)[iconv];
      reco::ConversionRef convref(singleConversionHandle,iconv);
      
      bool matched = ConversionTools::matchesConversion(gsfElectron,conversion,true,true);
      if (!matched) continue;
      
      if (!singleConversionMap.count(convref)) {
        singleConversions->push_back(conversion);
        singleConversionMap[convref] = singleConversions->size() - 1;
      }
      
    }    
    
  }
  
  //loop over output SuperClusters and fill maps
  for (unsigned int isc = 0; isc<superClusters->size(); ++isc) {
    reco::SuperCluster &superCluster = (*superClusters)[isc];
    
    //link seed cluster no matter what
    if (!ebeeClusterMap.count(superCluster.seed())) {
      ebeeClusters->push_back(*superCluster.seed());
      ebeeClusterMap[superCluster.seed()] = ebeeClusters->size() - 1;
    }
        
    //only proceed if superCluster is marked for full relinking
    bool fullrelink = superClusterFullRelinkMap.count(isc);
    if (!fullrelink) {
      //zero detid vector which is anyways not useful without stored rechits
      superCluster.clearHitsAndFractions();
      continue; 
    }
    
    for (const reco::CaloClusterPtr &cluster : superCluster.clusters()) {
      if (!ebeeClusterMap.count(cluster)) {
        ebeeClusters->push_back(*cluster);
        ebeeClusterMap[cluster] = ebeeClusters->size() - 1;
      }
      for (std::pair<DetId,float> hitfrac : cluster->hitsAndFractions()) {
        rechitMap.insert(hitfrac.first);
      }
      //make sure to also take all hits in the 5x5 around the max energy xtal
      bool barrel = cluster->hitsAndFractions().front().first.subdetId()==EcalBarrel;
      const EcalRecHitCollection *rhcol = barrel ? barrelHitHandle.product() : endcapHitHandle.product();
      DetId seed = EcalClusterTools::getMaximum(*cluster, rhcol).first;
      
      std::vector<DetId> dets5x5 = (barrel) ? caloTopology->getSubdetectorTopology(DetId::Ecal,EcalBarrel)->getWindow(seed,5,5) : caloTopology->getSubdetectorTopology(DetId::Ecal,EcalEndcap)->getWindow(seed,5,5);
      for (const DetId &detid : dets5x5) {
        rechitMap.insert(detid);
      }
    }
    for (const reco::CaloClusterPtr &cluster : superCluster.preshowerClusters()) {
      if (!esClusterMap.count(cluster)) {
        esClusters->push_back(*cluster);
        esClusterMap[cluster] = esClusters->size() - 1;
      }
      for (std::pair<DetId,float> hitfrac : cluster->hitsAndFractions()) {
        rechitMap.insert(hitfrac.first);
      }      
    }
    
    //conversions matched geometrically
    for (unsigned int iconv = 0; iconv<conversionHandle->size(); ++iconv) {
      const reco::Conversion &conversion = (*conversionHandle)[iconv];
      reco::ConversionRef convref(conversionHandle,iconv);
      
      bool matched = ConversionTools::matchesConversion(superCluster,conversion,0.2);
      if (!matched) continue;
      
      if (!conversionMap.count(convref)) {
        conversions->push_back(conversion);
        conversionMap[convref] = conversions->size() - 1;
      }
      
    }
    
    //single leg conversions matched by trackrefs
    for (unsigned int iconv = 0; iconv<singleConversionHandle->size(); ++iconv) {
      const reco::Conversion &conversion = (*singleConversionHandle)[iconv];
      reco::ConversionRef convref(singleConversionHandle,iconv);
      
      bool matched = ConversionTools::matchesConversion(superCluster,conversion,0.2);
      if (!matched) continue;
      
      if (!singleConversionMap.count(convref)) {
        singleConversions->push_back(conversion);
        singleConversionMap[convref] = singleConversions->size() - 1;
      }
      
    }
    
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
  
  for (const EcalRecHit &rechit : *preshowerHitHandle) {
    if (rechitMap.count(rechit.detid())) {
      esRecHits->push_back(rechit);
    }
  }
  
  theEvent.put(ebRecHits,outEBRecHits_);
  theEvent.put(eeRecHits,outEERecHits_);
  theEvent.put(esRecHits,outESRecHits_);  
  
  
  //CaloClusters
  //put calocluster output collections in event and get orphan handles to create ptrs
  const edm::OrphanHandle<reco::CaloClusterCollection> &outEBEEClusterHandle = theEvent.put(ebeeClusters,outEBEEClusters_);
  const edm::OrphanHandle<reco::CaloClusterCollection> &outESClusterHandle = theEvent.put(esClusters,outESClusters_);;  
  
  //loop over output superclusters and relink to output caloclusters
  for (reco::SuperCluster &superCluster : *superClusters) {
    //remap seed cluster
    const auto &seedmapped = ebeeClusterMap.find(superCluster.seed());
    if (seedmapped != ebeeClusterMap.end()) {
      //make new ptr
      reco::CaloClusterPtr clusptr(outEBEEClusterHandle,seedmapped->second);
      superCluster.setSeed(clusptr);
    }
    
    //remap all clusters
    reco::CaloClusterPtrVector clusters;
    for (const reco::CaloClusterPtr &cluster : superCluster.clusters()) {
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
    for (const reco::CaloClusterPtr &cluster : superCluster.preshowerClusters()) {
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
  
  //put superclusters and conversions in the event
  const edm::OrphanHandle<reco::SuperClusterCollection> &outSuperClusterHandle = theEvent.put(superClusters,outSuperClusters_);
  const edm::OrphanHandle<reco::ConversionCollection> &outConversionHandle = theEvent.put(conversions,outConversions_);
  const edm::OrphanHandle<reco::ConversionCollection> &outSingleConversionHandle = theEvent.put(singleConversions,outSingleConversions_);
  
  //loop over photoncores and relink superclusters (and conversions)
  for (reco::PhotonCore &photonCore : *photonCores) {
    const auto &scmapped = superClusterMap.find(photonCore.superCluster());
    if (scmapped != superClusterMap.end()) {
      //make new ref
      reco::SuperClusterRef scref(outSuperClusterHandle,scmapped->second);
      photonCore.setSuperCluster(scref);
    }
    
    //conversions
    const reco::ConversionRefVector &convrefs = photonCore.conversions();
    reco::ConversionRefVector outconvrefs;
    for (const reco::ConversionRef &convref : convrefs) {
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
    
    //single leg conversions
    const reco::ConversionRefVector &singleconvrefs = photonCore.conversionsOneLeg();
    reco::ConversionRefVector outsingleconvrefs;
    for (const reco::ConversionRef &convref : singleconvrefs) {
      const auto &convmapped = singleConversionMap.find(convref);
      if (convmapped != singleConversionMap.end()) {
        //make new ref
        reco::ConversionRef outref(outSingleConversionHandle,convmapped->second);
      }
      else {
        //can only relink if all conversions are being relinked, so if one is missing, then skip the relinking completely
        outsingleconvrefs.clear();
        break;
      }
    }
    if (outsingleconvrefs.size()) {
      photonCore.setConversionsOneLeg(outsingleconvrefs);
    }    
    
  }
  
  //loop over gsfelectroncores and relink superclusters
  for (reco::GsfElectronCore &gsfElectronCore : *gsfElectronCores) {
    const auto &scmapped = superClusterMap.find(gsfElectronCore.superCluster());
    if (scmapped != superClusterMap.end()) {
      //make new ref
      reco::SuperClusterRef scref(outSuperClusterHandle,scmapped->second);
      gsfElectronCore.setSuperCluster(scref);
    }
  }
  
  //put photon and gsfelectroncores into the event
  const edm::OrphanHandle<reco::PhotonCoreCollection> &outPhotonCoreHandle = theEvent.put(photonCores,outPhotonCores_);
  const edm::OrphanHandle<reco::GsfElectronCoreCollection> &outgsfElectronCoreHandle = theEvent.put(gsfElectronCores,outGsfElectronCores_);
  
  //loop over photons and electrons and relink the cores
  for (reco::Photon &photon : *photons) {
    const auto &coremapped = photonCoreMap.find(photon.photonCore());
    if (coremapped != photonCoreMap.end()) {
      //make new ref
      reco::PhotonCoreRef coreref(outPhotonCoreHandle,coremapped->second);
      photon.setPhotonCore(coreref);
    }
  }

  for (reco::GsfElectron &gsfElectron : *gsfElectrons) {
    const auto &coremapped = gsfElectronCoreMap.find(gsfElectron.core());
    if (coremapped != gsfElectronCoreMap.end()) {
      //make new ref
      reco::GsfElectronCoreRef coreref(outgsfElectronCoreHandle,coremapped->second);
      gsfElectron.setCore(coreref);
    }
  }
  
  //(finally) store the output photon and electron collections
  const edm::OrphanHandle<reco::PhotonCollection> &outPhotonHandle = theEvent.put(photons,outPhotons_);  
  const edm::OrphanHandle<reco::GsfElectronCollection> &outGsfElectronHandle = theEvent.put(gsfElectrons,outGsfElectrons_);
  
  //still need to output relinked valuemaps
  
  //photon pfcand isolation valuemap
  edm::ValueMap<std::vector<reco::PFCandidateRef>>::Filler fillerPhotons(*photonPfCandMap);
  fillerPhotons.insert(outPhotonHandle,pfCandIsoPairVecPho.begin(),pfCandIsoPairVecPho.end());
  fillerPhotons.fill();   
  
  //electron pfcand isolation valuemap
  edm::ValueMap<std::vector<reco::PFCandidateRef>>::Filler fillerGsfElectrons(*gsfElectronPfCandMap);
  fillerGsfElectrons.insert(outGsfElectronHandle,pfCandIsoPairVecEle.begin(),pfCandIsoPairVecEle.end());
  fillerGsfElectrons.fill();
  
  theEvent.put(photonPfCandMap,outPhotonPfCandMap_);
  theEvent.put(gsfElectronPfCandMap,outGsfElectronPfCandMap_);
  
  //photon id value maps
  for (unsigned int iid=0; iid<photonIds.size(); ++iid) {
    edm::ValueMap<bool>::Filler fillerPhotonId(*photonIds[iid]);
    fillerPhotonId.insert(outPhotonHandle,photonIdVals[iid].begin(),photonIdVals[iid].end());
    fillerPhotonId.fill();
    theEvent.put(photonIds[iid],outPhotonIds_[iid]);
  }
  
  //electron id value maps
  for (unsigned int iid=0; iid<gsfElectronIds.size(); ++iid) {
    edm::ValueMap<float>::Filler fillerGsfElectronId(*gsfElectronIds[iid]);
    fillerGsfElectronId.insert(outGsfElectronHandle,gsfElectronIdVals[iid].begin(),gsfElectronIdVals[iid].end());
    fillerGsfElectronId.fill();
    theEvent.put(gsfElectronIds[iid],outGsfElectronIds_[iid]);
  }  

  //photon iso value maps
  for (unsigned int iid=0; iid<photonPFClusterIsos.size(); ++iid) {
    edm::ValueMap<float>::Filler fillerPhotonPFClusterIso(*photonPFClusterIsos[iid]);
    fillerPhotonPFClusterIso.insert(outPhotonHandle,photonPFClusterIsoVals[iid].begin(),photonPFClusterIsoVals[iid].end());
    fillerPhotonPFClusterIso.fill();
    theEvent.put(photonPFClusterIsos[iid],outPhotonPFClusterIsos_[iid]);
  }
  //electron iso value maps
  for (unsigned int iid=0; iid<gsfElectronPFClusterIsos.size(); ++iid) {
    edm::ValueMap<float>::Filler fillerGsfElectronPFClusterIso(*gsfElectronPFClusterIsos[iid]);
    fillerGsfElectronPFClusterIso.insert(outGsfElectronHandle,gsfElectronPFClusterIsoVals[iid].begin(),gsfElectronPFClusterIsoVals[iid].end());
    fillerGsfElectronPFClusterIso.fill();
    theEvent.put(gsfElectronPFClusterIsos[iid],outGsfElectronPFClusterIsos_[iid]);
  }  
}


