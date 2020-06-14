
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/EgammaReco/interface/EgTrigSumObj.h"
#include "DataFormats/EgammaReco/interface/EgTrigSumObjFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateIsolation.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include <vector>

class EgammaHLTExtraProducer : public edm::global::EDProducer<> {
public:
  explicit EgammaHLTExtraProducer(const edm::ParameterSet& pset);
  ~EgammaHLTExtraProducer() override{}

  void produce(edm::StreamID streamID, edm::Event& event, const edm::EventSetup& eventSetup) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  
  static void setVars(reco::EgTrigSumObj& egTrigObj,const reco::RecoEcalCandidateRef& ecalCandRef,const std::vector<edm::Handle<reco::RecoEcalCandidateIsolationMap> >& valueMapHandles);
  static reco::GsfTrackRefVector matchingGsfTrks(const reco::SuperClusterRef& scRef,const edm::Handle<reco::GsfTrackCollection>& gsfTrksHandle);
  static void setGsfTracks(reco::EgTrigSumObj& egTrigObj,const edm::Handle<reco::GsfTrackCollection>& gsfTrksHandle);
  static void setSeeds(reco::EgTrigSumObj& egTrigObj,edm::Handle<reco::ElectronSeedCollection>& eleSeedsHandle);

  //these three filter functions are overly similar but with annoying differences
  //eg rechits needs to access geometry, trk dr is also w.r.t the track eta/phi
  //still could collapse into a single function
  template<typename RecHitCollection>
  std::unique_ptr<RecHitCollection> 
  filterRecHits(const reco::EgTrigSumObjCollection& egTrigObjs,
		const edm::Handle<RecHitCollection>& recHits,
		const CaloGeometry& geom,float maxDR2=0.4*0.4)const;

  std::unique_ptr<reco::TrackCollection>
  filterTrks(const reco::EgTrigSumObjCollection& egTrigObjs,const edm::Handle<reco::TrackCollection>& trks,float maxDR2=0.4*0.4)const;

  std::unique_ptr<reco::PFClusterCollection> 
  filterPFClusIso(const reco::EgTrigSumObjCollection& egTrigObjs,const edm::Handle<reco::PFClusterCollection>& pfClus,float maxDR2=0.4*0.4)const;

  struct Tokens {
    edm::EDGetTokenT<reco::RecoEcalCandidateCollection> ecalCands;
    edm::EDGetTokenT<reco::GsfTrackCollection> gsfTracks;
    edm::EDGetTokenT<reco::ElectronSeedCollection> pixelSeeds;
    std::vector<std::pair<edm::EDGetTokenT<EcalRecHitCollection>,std::string> > ecal;
    std::vector<std::pair<edm::EDGetTokenT<HBHERecHitCollection>,std::string> > hcal;
    std::vector<std::pair<edm::EDGetTokenT<reco::TrackCollection>,std::string> > trks;
    std::vector<std::pair<edm::EDGetTokenT<reco::PFClusterCollection>,std::string> > pfClusIso;
    
    template<typename T>
    static void setToken(edm::EDGetTokenT<T>& token,edm::ConsumesCollector& cc,const edm::ParameterSet& pset,const std::string& tagname){
      token = cc.consumes<T>(pset.getParameter<edm::InputTag>(tagname));
    }
    template<typename T>
    static void setToken(std::vector<edm::EDGetTokenT<T> >& tokens,edm::ConsumesCollector& cc,const edm::ParameterSet& pset,const std::string& tagname){
      auto inputTags = pset.getParameter<std::vector<edm::InputTag>>(tagname);     
      tokens.resize(inputTags.size());
      for(size_t tagNr=0;tagNr<inputTags.size();tagNr++){
     	tokens[tagNr] = cc.consumes<T>(inputTags[tagNr]);
      }
    }
    template<typename T>
    static void setToken(std::vector<std::pair<edm::EDGetTokenT<T>,std::string> >& tokens,edm::ConsumesCollector& cc,const edm::ParameterSet& pset,const std::string& tagname){
      const auto& collectionPSets = pset.getParameter<std::vector<edm::ParameterSet> >(tagname);
      for(const auto& collPSet : collectionPSets){
	edm::EDGetTokenT<T> token = cc.consumes<T>(collPSet.getParameter<edm::InputTag>("src"));
	std::string label = collPSet.getParameter<std::string>("label");
	tokens.emplace_back(std::make_pair(token,label));
      }
    }
    Tokens(const edm::ParameterSet& pset,edm::ConsumesCollector&& cc);
    
  };


  const Tokens tokens_;

  float minPtToSaveHits_;
  bool saveHitsPlusPi_;
  bool saveHitsPlusHalfPi_;
  
};

EgammaHLTExtraProducer::Tokens::Tokens(const edm::ParameterSet& pset,edm::ConsumesCollector&& cc)
{
  setToken(ecalCands,cc,pset,"ecalCands");
  setToken(pixelSeeds,cc,pset,"pixelSeeds");
  setToken(gsfTracks,cc,pset,"gsfTracks");
  setToken(ecal,cc,pset,"ecal");
  setToken(hcal,cc,pset,"hcal");
  setToken(trks,cc,pset,"trks");  
  setToken(pfClusIso,cc,pset,"pfClusIso");
}


EgammaHLTExtraProducer::EgammaHLTExtraProducer(const edm::ParameterSet& pset):
  tokens_(pset,consumesCollector()),
  minPtToSaveHits_(pset.getParameter<double>("minPtToSaveHits")),
  saveHitsPlusPi_(pset.getParameter<bool>("saveHitsPlusPi")),
  saveHitsPlusHalfPi_(pset.getParameter<bool>("saveHitsPlusHalfPi"))
{
  consumesMany<reco::RecoEcalCandidateIsolationMap>();

  produces<reco::EgTrigSumObjCollection>();
  for(auto& tokenLabel : tokens_.ecal){
    produces<EcalRecHitCollection>(tokenLabel.second);
  }
  for(auto& tokenLabel : tokens_.hcal){
    produces<HBHERecHitCollection>(tokenLabel.second);
  }
  for(auto& tokenLabel : tokens_.trks){
    produces<reco::TrackCollection>(tokenLabel.second);
  }
  for(auto& tokenLabel : tokens_.pfClusIso){
    produces<reco::PFClusterCollection>(tokenLabel.second);
  }
  
}

void EgammaHLTExtraProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("ecalCands", edm::InputTag("ecalCands"));
  desc.add<edm::InputTag>("pixelSeeds", edm::InputTag("pixelSeeds"));
  desc.add<edm::InputTag>("gsfTracks", edm::InputTag("gsfTracks"));  
  desc.add<double>("minPtToSaveHits",0.);
  desc.add<bool>("saveHitsPlusPi",true);
  desc.add<bool>("saveHitsPlusHalfPi",true);

  edm::ParameterSetDescription tokenLabelDesc;
  tokenLabelDesc.add<edm::InputTag>("src",edm::InputTag(""));
  tokenLabelDesc.add<std::string>("label","");
  std::vector<edm::ParameterSet> ecalDefaults(2);
  ecalDefaults[0].addParameter("src",edm::InputTag("hltEcalRecHit","EcalRecHitEB"));
  ecalDefaults[0].addParameter("label",std::string("EcalRecHitsEB"));
  ecalDefaults[1].addParameter("src",edm::InputTag("hltEcalRecHit","EcalRecHitEE"));
  ecalDefaults[1].addParameter("label",std::string("EcalRecHitsEE"));
  std::vector<edm::ParameterSet> hcalDefaults(1);
  hcalDefaults[0].addParameter("src",edm::InputTag("hltHbhereco"));
  hcalDefaults[0].addParameter("label",std::string(""));
  std::vector<edm::ParameterSet> trksDefaults(1);
  trksDefaults[0].addParameter("src",edm::InputTag("generalTracks"));
  trksDefaults[0].addParameter("label",std::string(""));
  std::vector<edm::ParameterSet> pfClusIsoDefaults(2);
  pfClusIsoDefaults[0].addParameter("src",edm::InputTag("hltParticleFlowClusterECALUnseeded"));
  pfClusIsoDefaults[0].addParameter("label",std::string("Ecal"));
  pfClusIsoDefaults[1].addParameter("src",edm::InputTag("hltParticleFlowClusterHCAL"));
  pfClusIsoDefaults[1].addParameter("label",std::string("Hcal"));

  desc.addVPSet("ecal",tokenLabelDesc,ecalDefaults);
  desc.addVPSet("hcal",tokenLabelDesc,hcalDefaults);
  desc.addVPSet("trks",tokenLabelDesc,trksDefaults);
  desc.addVPSet("pfClusIso",tokenLabelDesc,pfClusIsoDefaults);
  
  descriptions.add(("hltEgammaHLTExtraProducer"), desc);
}

void EgammaHLTExtraProducer::produce(edm::StreamID streamID,
				     edm::Event& event,
				     const edm::EventSetup& eventSetup) const {
  auto ecalCandsHandle = event.getHandle(tokens_.ecalCands);
  auto gsfTrksHandle = event.getHandle(tokens_.gsfTracks);
  auto pixelSeedsHandle = event.getHandle(tokens_.pixelSeeds);
  
  std::vector<edm::Handle<reco::RecoEcalCandidateIsolationMap> > valueMapHandles;
  event.getManyByType(valueMapHandles);

  auto egTrigObjs = std::make_unique<reco::EgTrigSumObjCollection>();
  for(size_t candNr=0;ecalCandsHandle.isValid() && candNr<ecalCandsHandle->size();candNr++){
    reco::RecoEcalCandidateRef candRef(ecalCandsHandle,candNr);
    egTrigObjs->push_back(*candRef);
    auto& egTrigObj = egTrigObjs->back();
    setVars(egTrigObj,candRef,valueMapHandles);
    setGsfTracks(egTrigObj,gsfTrksHandle);
    setSeeds(egTrigObj,pixelSeedsHandle);
  }

  edm::ESHandle<CaloGeometry> caloGeomHandle;
  eventSetup.get<CaloGeometryRecord>().get(caloGeomHandle);

  auto filterAndStoreRecHits=[caloGeomHandle,&event,this](const reco::EgTrigSumObjCollection& egTrigObjs,auto& tokenLabels){
    for(const auto& tokenLabel : tokenLabels){
      auto handle = event.getHandle(tokenLabel.first);
      auto recHits = filterRecHits(egTrigObjs,handle,*caloGeomHandle);
      event.put(std::move(recHits),tokenLabel.second);
    }
  };
  auto filterAndStore=[&event,this](const reco::EgTrigSumObjCollection& egTrigObjs,auto& tokenLabels,auto filterFunc){
    for(const auto& tokenLabel : tokenLabels){
      auto handle = event.getHandle(tokenLabel.first);
      auto filtered = (this->*filterFunc)(egTrigObjs,handle,0.4*0.4);
      event.put(std::move(filtered),tokenLabel.second);
    }
  };

  filterAndStoreRecHits(*egTrigObjs,tokens_.ecal);
  filterAndStoreRecHits(*egTrigObjs,tokens_.hcal);
  filterAndStore(*egTrigObjs,tokens_.pfClusIso,&EgammaHLTExtraProducer::filterPFClusIso);
  filterAndStore(*egTrigObjs,tokens_.trks,&EgammaHLTExtraProducer::filterTrks);

  // for(const auto& tokenLabel : tokens_.trks){
  //   auto handle = event.getHandle(tokenLabel.first);
  //   auto trks = filterTrks(*egTrigObjs,handle);
  //   event.put(std::move(trks),tokenLabel.second);
  // }
  
    
  event.put(std::move(egTrigObjs));
  
}

void EgammaHLTExtraProducer::setVars(reco::EgTrigSumObj& egTrigObj,const reco::RecoEcalCandidateRef& ecalCandRef,const std::vector<edm::Handle<reco::RecoEcalCandidateIsolationMap> >& valueMapHandles)
{
  for(auto& valueMapHandle : valueMapHandles){
    auto mapIt = valueMapHandle->find(ecalCandRef);
    if(mapIt!=valueMapHandle->end()){
      std::string name=valueMapHandle.provenance()->moduleLabel();
      if(!valueMapHandle.provenance()->productInstanceName().empty()){
	name+="_"+valueMapHandle.provenance()->productInstanceName();
      }
      egTrigObj.setVar(std::move(name),mapIt->val);
    }
  }
}

reco::GsfTrackRefVector EgammaHLTExtraProducer::matchingGsfTrks(const reco::SuperClusterRef& scRef,const edm::Handle<reco::GsfTrackCollection>& gsfTrksHandle)
{
  if(!gsfTrksHandle.isValid()){
    return reco::GsfTrackRefVector();
  }

  reco::GsfTrackRefVector gsfTrkRefs(gsfTrksHandle.id());
  for(size_t trkNr=0;gsfTrksHandle.isValid() && trkNr<gsfTrksHandle->size();trkNr++){
    reco::GsfTrackRef trkRef(gsfTrksHandle,trkNr);
    edm::RefToBase<TrajectorySeed> seed = trkRef->extra()->seedRef();
    reco::ElectronSeedRef eleSeed = seed.castTo<reco::ElectronSeedRef>();
    edm::RefToBase<reco::CaloCluster> caloCluster = eleSeed->caloCluster();
    reco::SuperClusterRef scRefFromTrk = caloCluster.castTo<reco::SuperClusterRef>();
    if( scRefFromTrk == scRef){
      gsfTrkRefs.push_back(trkRef);
    }
  }
  return gsfTrkRefs;
}

void EgammaHLTExtraProducer::setGsfTracks(reco::EgTrigSumObj& egTrigObj,const edm::Handle<reco::GsfTrackCollection>& gsfTrksHandle)
{
  reco::GsfTrackRefVector gsfTrkRefs = matchingGsfTrks(egTrigObj.superCluster(),gsfTrksHandle);
  egTrigObj.setGsfTracks(std::move(gsfTrkRefs));
}

void EgammaHLTExtraProducer::setSeeds(reco::EgTrigSumObj& egTrigObj,edm::Handle<reco::ElectronSeedCollection>& eleSeedsHandle)
{
  if(!eleSeedsHandle.isValid()){
    egTrigObj.setSeeds(reco::ElectronSeedRefVector());
  }else{
    reco::ElectronSeedRefVector trigObjSeeds(eleSeedsHandle.id());
    
    
    for(size_t seedNr=0;eleSeedsHandle.isValid() && seedNr<eleSeedsHandle->size();seedNr++){
      
      reco::ElectronSeedRef eleSeed(eleSeedsHandle,seedNr);
      edm::RefToBase<reco::CaloCluster> caloCluster = eleSeed->caloCluster();
      reco::SuperClusterRef scRefFromSeed = caloCluster.castTo<reco::SuperClusterRef>();
      
      if(scRefFromSeed==egTrigObj.superCluster()){
	trigObjSeeds.push_back(eleSeed);
      }
    }
    egTrigObj.setSeeds(std::move(trigObjSeeds));
  }
}


template<typename RecHitCollection>
std::unique_ptr<RecHitCollection> EgammaHLTExtraProducer::filterRecHits(const reco::EgTrigSumObjCollection& egTrigObjs,const edm::Handle<RecHitCollection>& recHits,const CaloGeometry& geom,float maxDR2)const
{
  auto filteredHits = std::make_unique<RecHitCollection>();
  if(!recHits.isValid()) return filteredHits;

  std::vector<std::pair<float,float> > etaPhis;
  for(const auto& egTrigObj : egTrigObjs){
    if(egTrigObj.pt()>=minPtToSaveHits_){
      etaPhis.push_back({egTrigObj.eta(),egTrigObj.phi()});
      if(saveHitsPlusPi_) etaPhis.push_back({egTrigObj.eta(),egTrigObj.phi()+3.14159});
      if(saveHitsPlusHalfPi_) etaPhis.push_back({egTrigObj.eta(),egTrigObj.phi()+3.14159/2.});
    }
  }
  auto deltaR2Match = [&etaPhis,&maxDR2](const GlobalPoint& pos){
    float eta = pos.eta();
    float phi = pos.phi();
    for(auto& etaPhi : etaPhis){ 
      if(reco::deltaR2(eta,phi,etaPhi.first,etaPhi.second)<maxDR2) return true;
    }
    return false;
  };

  for(auto& hit : *recHits){
    const CaloSubdetectorGeometry* subDetGeom =  geom.getSubdetectorGeometry(hit.id());
    if(subDetGeom){
      auto cellGeom = subDetGeom->getGeometry(hit.id());
      if(deltaR2Match(cellGeom->getPosition())) filteredHits->push_back(hit);
    }else{
      throw cms::Exception("GeomError") << "could not get geometry for det id "<<hit.id().rawId();
    }
  }
  return filteredHits;
}

std::unique_ptr<reco::TrackCollection> EgammaHLTExtraProducer::filterTrks(const reco::EgTrigSumObjCollection& egTrigObjs,const edm::Handle<reco::TrackCollection>& trks,float maxDR2)const
{
  auto filteredTrks = std::make_unique<reco::TrackCollection>();
  if(!trks.isValid()) return filteredTrks;

  //so because each egamma object can have multiple eta/phi pairs
  //easier to just make a temp vector and then copy that in with the +pi and  +pi/2
  std::vector<std::pair<float,float> > etaPhisTmp;
  for(const auto& egTrigObj : egTrigObjs){
    if(egTrigObj.pt()>=minPtToSaveHits_){
      etaPhisTmp.push_back({egTrigObj.eta(),egTrigObj.phi()});
      //also save the eta /phi of all gsf tracks with the object
      for(const auto& gsfTrk : egTrigObj.gsfTracks()){
	etaPhisTmp.push_back({gsfTrk->eta(),gsfTrk->phi()});
      }
    }
  }
  std::vector<std::pair<float,float> > etaPhis;
  for(const auto& etaPhi : etaPhisTmp){
    etaPhis.push_back(etaPhi);
    if(saveHitsPlusPi_) etaPhis.push_back({etaPhi.first,etaPhi.second+3.14159});
    if(saveHitsPlusHalfPi_) etaPhis.push_back({etaPhi.first,etaPhi.second+3.14159/2.});
  }

  auto deltaR2Match = [&etaPhis,&maxDR2](float eta,float phi){
    for(auto& etaPhi : etaPhis){ 
      if(reco::deltaR2(eta,phi,etaPhi.first,etaPhi.second)<maxDR2) return true;
    }
    return false;
  };

  for(auto& trk : *trks){
    if(deltaR2Match(trk.eta(),trk.phi())) filteredTrks->push_back(trk);
  }
  return filteredTrks;
}

std::unique_ptr<reco::PFClusterCollection> EgammaHLTExtraProducer::filterPFClusIso(const reco::EgTrigSumObjCollection& egTrigObjs,const edm::Handle<reco::PFClusterCollection>& pfClus,float maxDR2)const
{
  auto filteredPFClus = std::make_unique<reco::PFClusterCollection>();
  if(!pfClus.isValid()) return filteredPFClus;
  
  std::vector<std::pair<float,float> > etaPhis;
  for(const auto& egTrigObj : egTrigObjs){
    if(egTrigObj.pt()>=minPtToSaveHits_){
      etaPhis.push_back({egTrigObj.eta(),egTrigObj.phi()});
      if(saveHitsPlusPi_) etaPhis.push_back({egTrigObj.eta(),egTrigObj.phi()+3.14159});
      if(saveHitsPlusHalfPi_) etaPhis.push_back({egTrigObj.eta(),egTrigObj.phi()+3.14159/2.});
    }
  }
  auto deltaR2Match = [&etaPhis,&maxDR2](float eta,float phi){
    for(auto& etaPhi : etaPhis){ 
      if(reco::deltaR2(eta,phi,etaPhi.first,etaPhi.second)<maxDR2) return true;
    }
    return false;
  };

  for(auto& clus : *pfClus){
    if(deltaR2Match(clus.eta(),clus.phi())) filteredPFClus->push_back(clus);
  }
  return filteredPFClus;
}


#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(EgammaHLTExtraProducer);
