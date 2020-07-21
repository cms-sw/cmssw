
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
#include "DataFormats/L1TrackTrigger/interface/TTTrack.h"
#include "DataFormats/L1TrackTrigger/interface/TTTrackExtra.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Common/interface/RefToPtr.h"

#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTTrackAssociationMap.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include <vector>
#include <unordered_map>

class EgammaHLTPhase2ExtraProducer : public edm::global::EDProducer<> {
public:
  
  //normally I hate typedefs but in this case, TTrackTrigger should have typedefed this themselves
  using L1Track = TTTrack<Ref_Phase2TrackerDigi_>;
  using L1TrackCollection = std::vector<L1Track>;
  using L1TrackExtraCollection = std::vector<L1TrackExtra>;

  explicit EgammaHLTPhase2ExtraProducer(const edm::ParameterSet& pset);
  ~EgammaHLTPhase2ExtraProducer() override{}

  void produce(edm::StreamID streamID, edm::Event& event, const edm::EventSetup& eventSetup) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  

  template<typename CollType,typename RefType>
  std::unique_ptr<CollType> filterObjs(const reco::EgTrigSumObjCollection& egTrigObjs,const edm::Handle<CollType>& objs,std::vector<RefType>& orgRefs,float maxDR2=0.4*0.4)const;

  struct Tokens {
    edm::EDGetTokenT<reco::EgTrigSumObjCollection> egTrigObjs;
    edm::EDGetTokenT<L1TrackCollection> l1Trks;
    edm::EDGetTokenT<TrackingParticleCollection> trkParts;
    edm::EDGetTokenT<TTTrackAssociationMap<Ref_Phase2TrackerDigi_> > l1TrkToTrkPartMap;

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

EgammaHLTPhase2ExtraProducer::Tokens::Tokens(const edm::ParameterSet& pset,edm::ConsumesCollector&& cc)
{
  setToken(egTrigObjs,cc,pset,"egTrigObjs");
  setToken(l1Trks,cc,pset,"l1Trks");
  setToken(trkParts,cc,pset,"trkParts");
  setToken(l1TrkToTrkPartMap,cc,pset,"l1TrkToTrkPartMap");
}


EgammaHLTPhase2ExtraProducer::EgammaHLTPhase2ExtraProducer(const edm::ParameterSet& pset):
  tokens_(pset,consumesCollector()),
  minPtToSaveHits_(pset.getParameter<double>("minPtToSaveHits")),
  saveHitsPlusPi_(pset.getParameter<bool>("saveHitsPlusPi")),
  saveHitsPlusHalfPi_(pset.getParameter<bool>("saveHitsPlusHalfPi"))
{
  produces<L1TrackCollection>();
  produces<L1TrackExtraCollection>();
  produces<TrackingParticleCollection>();
}

void EgammaHLTPhase2ExtraProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("egTrigObjs", edm::InputTag("hltEgammaHLTExtra"));
  desc.add<edm::InputTag>("l1Trks", edm::InputTag("TTTracksFromTrackletEmulation","Level1TTTracks"));
  desc.add<edm::InputTag>("trkParts", edm::InputTag("mix","MergedTrackTruth"));  
  desc.add<edm::InputTag>("l1TrkToTrkPartMap", edm::InputTag("TTTrackAssociatorFromPixelDigis","Level1TTTracks"));  
  desc.add<double>("minPtToSaveHits",0.);
  desc.add<bool>("saveHitsPlusPi",true);
  desc.add<bool>("saveHitsPlusHalfPi",true);
  descriptions.add(("hltEgammaHLTPhase2ExtraProducer"), desc);
}

void EgammaHLTPhase2ExtraProducer::produce(edm::StreamID streamID,
				     edm::Event& event,
				     const edm::EventSetup& eventSetup) const {
  auto egTrigObjs = event.getHandle(tokens_.egTrigObjs);
      
  auto trkParts = event.getHandle(tokens_.trkParts);
  auto l1trks = event.getHandle(tokens_.l1Trks);
  auto l1TrkToTrkPartMap = event.getHandle(tokens_.l1TrkToTrkPartMap);
  
  std::vector<edm::Ref<L1TrackCollection> > orgL1TrkRefs;
  auto l1TrksFiltered = filterObjs(*egTrigObjs,l1trks,orgL1TrkRefs);
  std::vector<edm::Ref<TrackingParticleCollection> > orgTPRefs;
  auto trkPartsFiltered = filterObjs(*egTrigObjs,trkParts,orgTPRefs);

  //meh should make this edm::Ref<T>::key_type
  std::unordered_map<size_t,size_t> orgTPIndxToNewIndx;
  for(size_t refNr=0;refNr<orgTPRefs.size();refNr++){
    orgTPIndxToNewIndx.insert(std::make_pair(orgTPRefs[refNr].key(),refNr));
  }
  
  edm::OrphanHandle<L1TrackCollection> l1TrksFiltHandle = event.put(std::move(l1TrksFiltered));
  edm::OrphanHandle<TrackingParticleCollection> trkPartsFiltHandle = event.put(std::move(trkPartsFiltered));
  
  auto l1TrkExtraColl = std::make_unique<L1TrackExtraCollection>();
 
  for(size_t l1TrkNr=0;l1TrkNr<orgL1TrkRefs.size();l1TrkNr++){
    auto orgTrkRef = orgL1TrkRefs[l1TrkNr];
    auto orgTrkPtr = edm::refToPtr(orgTrkRef);
    int flags = 0;
    if(l1TrkToTrkPartMap->isGenuine(orgTrkPtr)) flags|=L1TrackExtra::StatusFlags::IsGenuine;
    if(l1TrkToTrkPartMap->isLooselyGenuine(orgTrkPtr)) flags|=L1TrackExtra::StatusFlags::IsLooselyGenuine;
    if(l1TrkToTrkPartMap->isCombinatoric(orgTrkPtr)) flags|=L1TrackExtra::StatusFlags::IsCombinatoric;   
    if(l1TrkToTrkPartMap->isUnknown(orgTrkPtr)) flags|=L1TrackExtra::StatusFlags::IsUnknown;


    auto orgTPRef = l1TrkToTrkPartMap->findTrackingParticlePtr(orgTrkPtr);
    auto getNewTPRef = [&orgTPIndxToNewIndx,&orgTPRef,&trkPartsFiltHandle](){
      auto newIndexPair = orgTPIndxToNewIndx.find(orgTPRef.key());
      if(newIndexPair!=orgTPIndxToNewIndx.end()){
	return edm::Ref<TrackingParticleCollection>(trkPartsFiltHandle,newIndexPair->second);
      }else return edm::Ref<TrackingParticleCollection>(trkPartsFiltHandle.id());
    };
    auto newTPRef = getNewTPRef();
    edm::Ref<L1TrackCollection> newL1TrkRef(l1TrksFiltHandle,l1TrkNr);

    L1TrackExtra l1TrkExtra(newL1TrkRef,newTPRef,flags);
    l1TrkExtraColl->push_back(l1TrkExtra);
  }
  event.put(std::move(l1TrkExtraColl));
  
}

template<typename CollType,typename RefType>
std::unique_ptr<CollType> EgammaHLTPhase2ExtraProducer::filterObjs(const reco::EgTrigSumObjCollection& egTrigObjs,const edm::Handle<CollType>& objs,std::vector<RefType>& orgRefs,float maxDR2)const
{
  auto filteredObjs = std::make_unique<CollType>();
  orgRefs.clear();
  if(!objs.isValid()) return filteredObjs;

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

  for(size_t objNr=0;objNr<objs->size();objNr++){
    RefType ref(objs,objNr);
    if(deltaR2Match(ref->eta(),ref->phi())){
      filteredObjs->push_back(*ref);
      orgRefs.push_back(ref);
    }
  }
  return filteredObjs;
}


#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(EgammaHLTPhase2ExtraProducer);
