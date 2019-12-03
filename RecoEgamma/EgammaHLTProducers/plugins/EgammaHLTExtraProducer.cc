
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

  template<typename RecHitCollection>
  static std::unique_ptr<RecHitCollection> 
  filterRecHits(const reco::EgTrigSumObjCollection& egTrigObjs,
		const edm::Handle<RecHitCollection>& recHits,
		const CaloGeometry& geom,float maxDR2=0.4*0.4);
 

  struct Tokens {
    edm::EDGetTokenT<reco::RecoEcalCandidateCollection> ecalCands;
    edm::EDGetTokenT<reco::GsfTrackCollection> gsfTracks;
    edm::EDGetTokenT<reco::ElectronSeedCollection> pixelSeeds;
    edm::EDGetTokenT<EcalRecHitCollection> ebRecHits;
    edm::EDGetTokenT<EcalRecHitCollection> eeRecHits;
    edm::EDGetTokenT<HBHERecHitCollection> hbheRecHits;

    template<typename T>
    void setToken(edm::EDGetTokenT<T>& token,edm::ConsumesCollector& cc,const edm::ParameterSet& pset,const std::string& tagname){
      token = cc.consumes<T>(pset.getParameter<edm::InputTag>(tagname));
    };
    Tokens(const edm::ParameterSet& pset,edm::ConsumesCollector&& cc);
    
  };
  const Tokens tokens_;
  
};

EgammaHLTExtraProducer::Tokens::Tokens(const edm::ParameterSet& pset,edm::ConsumesCollector&& cc)
{
  setToken(ecalCands,cc,pset,"ecalCands");
  setToken(ebRecHits,cc,pset,"ebRecHits");
  setToken(eeRecHits,cc,pset,"eeRecHits");
  setToken(hbheRecHits,cc,pset,"hbheRecHits");
  setToken(pixelSeeds,cc,pset,"pixelSeeds");
  setToken(gsfTracks,cc,pset,"gsfTracks");
}

EgammaHLTExtraProducer::EgammaHLTExtraProducer(const edm::ParameterSet& pset):
  tokens_(pset,consumesCollector())
{
  consumesMany<reco::RecoEcalCandidateIsolationMap>();

  produces<reco::EgTrigSumObjCollection>();
  produces<EcalRecHitCollection>("EcalRecHitsEB");
  produces<EcalRecHitCollection>("EcalRecHitsEE");
  produces<HBHERecHitCollection>();
}

void EgammaHLTExtraProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("ecalCands", edm::InputTag("ecalCands"));
  desc.add<edm::InputTag>("ebRecHits", edm::InputTag("ebRecHits"));
  desc.add<edm::InputTag>("eeRecHits", edm::InputTag("eeRecHits"));
  desc.add<edm::InputTag>("hbheRecHits", edm::InputTag("hbheRecHits"));
  desc.add<edm::InputTag>("pixelSeeds", edm::InputTag("pixelSeeds"));
  desc.add<edm::InputTag>("gsfTracks", edm::InputTag("gsfTracks"));

  descriptions.add(("hltEgammaHLTExtraProducer"), desc);
}

void EgammaHLTExtraProducer::produce(edm::StreamID streamID,
				     edm::Event& event,
				     const edm::EventSetup& eventSetup) const {

  auto ecalCandsHandle = event.getHandle(tokens_.ecalCands);
  auto gsfTrksHandle = event.getHandle(tokens_.gsfTracks);
  auto pixelSeedsHandle = event.getHandle(tokens_.pixelSeeds);
  auto ebRecHitsHandle = event.getHandle(tokens_.ebRecHits);
  auto eeRecHitsHandle = event.getHandle(tokens_.eeRecHits);
  auto hbheRecHitsHandle = event.getHandle(tokens_.hbheRecHits);

  std::vector<edm::Handle<reco::RecoEcalCandidateIsolationMap> > valueMapHandles;
  event.getManyByType(valueMapHandles);

  auto egTrigObjs = std::make_unique<reco::EgTrigSumObjCollection>();
  for(size_t candNr=0;candNr<ecalCandsHandle->size();candNr++){
    reco::RecoEcalCandidateRef candRef(ecalCandsHandle,candNr);
    egTrigObjs->push_back(*candRef);
    auto& egTrigObj = egTrigObjs->back();
    setVars(egTrigObj,candRef,valueMapHandles);
    setGsfTracks(egTrigObj,gsfTrksHandle);
    setSeeds(egTrigObj,pixelSeedsHandle);
  }

  edm::ESHandle<CaloGeometry> caloGeomHandle;
  eventSetup.get<CaloGeometryRecord>().get(caloGeomHandle);

  auto ebRecHitsFiltered = filterRecHits(*egTrigObjs,ebRecHitsHandle,*caloGeomHandle);
  auto eeRecHitsFiltered = filterRecHits(*egTrigObjs,eeRecHitsHandle,*caloGeomHandle);
  auto hbheRecHitsFiltered = filterRecHits(*egTrigObjs,hbheRecHitsHandle,*caloGeomHandle);
  
  event.put(std::move(egTrigObjs));
  event.put(std::move(ebRecHitsFiltered),"EcalRecHitsEB");
  event.put(std::move(eeRecHitsFiltered),"EcalRecHitsEE");
  event.put(std::move(hbheRecHitsFiltered));
  
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


template<typename RecHitCollection>
std::unique_ptr<RecHitCollection> EgammaHLTExtraProducer::filterRecHits(const reco::EgTrigSumObjCollection& egTrigObjs,const edm::Handle<RecHitCollection>& recHits,const CaloGeometry& geom,float maxDR2)
{
  auto filteredHits = std::make_unique<RecHitCollection>();
  if(!recHits.isValid()) return filteredHits;

  std::vector<std::pair<float,float> > etaPhis;
  for(const auto& egTrigObj : egTrigObjs){
    etaPhis.push_back({egTrigObj.eta(),egTrigObj.phi()});
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


#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(EgammaHLTExtraProducer);
