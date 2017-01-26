#ifndef RecoEcal_EgammaClusterProducers_MapNewToOldSCs_h
#define RecoEcal_EgammaClusterProducers_MapNewToOldSCs_h

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "RecoEgamma/EgammaTools/interface/GainSwitchTools.h"

//this allows us to make the old superclusters to the new superclusters
//via a valuemap
class MapNewToOldSCs : public edm::stream::EDProducer<> {
public:
  explicit MapNewToOldSCs(const edm::ParameterSet& );
  virtual ~MapNewToOldSCs(){}
  virtual void produce(edm::Event &, const edm::EventSetup &);
  
  static void writeSCRefValueMap(edm::Event &iEvent,
				 const edm::Handle<reco::SuperClusterCollection> & handle,
				 const std::vector<reco::SuperClusterRef> & values,
				 const std::string& label);
private:
  edm::EDGetTokenT<reco::SuperClusterCollection> oldRefinedSCToken_;
  edm::EDGetTokenT<reco::SuperClusterCollection> oldSCToken_;
  edm::EDGetTokenT<reco::SuperClusterCollection> newSCToken_;
  edm::EDGetTokenT<reco::SuperClusterCollection> newRefinedSCToken_;

  
  

};

MapNewToOldSCs::MapNewToOldSCs(const edm::ParameterSet& iConfig )
{
  oldRefinedSCToken_ = consumes<reco::SuperClusterCollection>(iConfig.getParameter<edm::InputTag>("oldRefinedSC"));
  oldSCToken_ = consumes<reco::SuperClusterCollection>(iConfig.getParameter<edm::InputTag>("oldSC"));
  newSCToken_ = consumes<reco::SuperClusterCollection>(iConfig.getParameter<edm::InputTag>("newSC"));
  newRefinedSCToken_ = consumes<reco::SuperClusterCollection>(iConfig.getParameter<edm::InputTag>("newRefinedSC"));

  produces<edm::ValueMap<reco::SuperClusterRef> >("parentSCs");
  produces<edm::ValueMap<reco::SuperClusterRef> >("refinedSCs");
  
}

namespace {
  template<typename T> edm::Handle<T> getHandle(const edm::Event& iEvent,const edm::EDGetTokenT<T>& token){
    edm::Handle<T> handle;
    iEvent.getByToken(token,handle);
    return handle;
  }
}

void MapNewToOldSCs::produce(edm::Event & iEvent, const edm::EventSetup & iSetup)
{
  auto newRefinedSCs = getHandle(iEvent,newRefinedSCToken_);
  auto newSCs = getHandle(iEvent,newSCToken_);
 
  auto oldRefinedSCs = getHandle(iEvent,oldRefinedSCToken_);
  auto oldSCs = getHandle(iEvent,oldSCToken_);
  
  std::vector<reco::SuperClusterRef> matchedNewSCs;
  for(auto oldSC : *oldSCs){
    matchedNewSCs.push_back(GainSwitchTools::matchSCBySeedCrys(oldSC,newSCs,2,2));
  }
  std::vector<reco::SuperClusterRef> matchedNewRefinedSCs;
  for(auto oldSC : *oldRefinedSCs){
    matchedNewRefinedSCs.push_back(GainSwitchTools::matchSCBySeedCrys(oldSC,newRefinedSCs,2,2));
  }
  
  writeSCRefValueMap(iEvent,oldSCs,matchedNewSCs,"parentSCs");
  writeSCRefValueMap(iEvent,oldRefinedSCs,matchedNewRefinedSCs,"refinedSCs");
  
}

void MapNewToOldSCs::writeSCRefValueMap(edm::Event &iEvent,
					  const edm::Handle<reco::SuperClusterCollection> & handle,
					  const std::vector<reco::SuperClusterRef> & values,
					  const std::string& label)
{ 
  std::unique_ptr<edm::ValueMap<reco::SuperClusterRef> > valMap(new edm::ValueMap<reco::SuperClusterRef>());
  typename edm::ValueMap<reco::SuperClusterRef>::Filler filler(*valMap);
  filler.insert(handle, values.begin(), values.end());
  filler.fill();
  iEvent.put(std::move(valMap),label);
}

DEFINE_FWK_MODULE(MapNewToOldSCs);

#endif
