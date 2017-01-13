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

namespace {
  int calDIEta(int lhs,int rhs)
  {
    int retVal = lhs - rhs;
    if(lhs*rhs<0){ //crossing zero
      if(retVal<0) retVal++;
      else retVal--;
    }
    return retVal;
  }

  int calDIPhi(int lhs,int rhs)
  {
    int retVal = lhs-rhs;
    while(retVal>180) retVal-=360;
    while(retVal<-180) retVal+=360;
    return retVal;
  }
  
  reco::SuperClusterRef matchSCBySeedCrys(const reco::SuperCluster& sc,edm::Handle<reco::SuperClusterCollection> scColl,int maxDEta,int maxDPhi)
  {
    reco::SuperClusterRef bestRef(scColl.id());
    
    int bestDIR2 = maxDEta*maxDEta+maxDPhi*maxDPhi+1; //+1 is to make it slightly bigger than max allowed
    
    if(sc.seed()->seed().subdetId()==EcalBarrel){
      EBDetId scDetId(sc.seed()->seed());
      
      for(size_t scNr=0;scNr<scColl->size();scNr++){
	reco::SuperClusterRef matchRef(scColl,scNr);
	if(matchRef->seed()->seed().subdetId()==EcalBarrel){
	  EBDetId matchDetId(matchRef->seed()->seed());
	  int dIEta = calDIEta(scDetId.ieta(),matchDetId.ieta());
	  int dIPhi = calDIPhi(scDetId.iphi(),matchDetId.iphi());
	  int dIR2 = dIEta*dIEta+dIPhi*dIPhi;
	  if(dIR2<bestDIR2){
	    bestDIR2=dIR2;
	    bestRef = reco::SuperClusterRef(scColl,scNr);
	  }
	}
      }     
    }
    return bestRef;
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
    matchedNewSCs.push_back(matchSCBySeedCrys(oldSC,newSCs,2,2));
  }
  std::vector<reco::SuperClusterRef> matchedNewRefinedSCs;
  for(auto oldSC : *oldRefinedSCs){
    matchedNewRefinedSCs.push_back(matchSCBySeedCrys(oldSC,newRefinedSCs,2,2));
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
