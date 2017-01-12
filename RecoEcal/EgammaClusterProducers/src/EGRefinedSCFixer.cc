#ifndef RecoEcal_EgammaClusterProducers_EGRefinedSCFixed_h
#define RecoEcal_EgammaClusterProducers_EGRefinedSCFixed_h

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"

class EGRefinedSCFixer : public edm::stream::EDProducer<> {
public:
  explicit EGRefinedSCFixer(const edm::ParameterSet& );
  virtual ~EGRefinedSCFixer(){}
  virtual void produce(edm::Event &, const edm::EventSetup &);
  
  reco::SuperCluster 
  makeFixedRefinedSC(const reco::SuperCluster& orgRefinedSC,
		     const reco::SuperCluster& orgSC,
		     const reco::SuperCluster& fixedSC);
		     

  static std::vector<edm::Ptr<reco::CaloCluster> >
  getSubClustersMissing(const reco::SuperCluster& lhs,
			const reco::SuperCluster& rhs);
    

private:
  edm::EDGetTokenT<reco::SuperClusterCollection> orgRefinedSCToken_;
  edm::EDGetTokenT<reco::SuperClusterCollection> orgSCToken_;
  edm::EDGetTokenT<reco::SuperClusterCollection> fixedSCToken_;
  
  
  

};

EGRefinedSCFixer::EGRefinedSCFixer(const edm::ParameterSet& iConfig )
{
  orgRefinedSCToken_ = consumes<reco::SuperClusterCollection>(iConfig.getParameter<edm::InputTag>("orgRefinedSC"));
  orgSCToken_ = consumes<reco::SuperClusterCollection>(iConfig.getParameter<edm::InputTag>("orgSC"));
  fixedSCToken_ = consumes<reco::SuperClusterCollection>(iConfig.getParameter<edm::InputTag>("fixedSC"));
}

namespace {
  template<typename T> edm::Handle<T> getHandle(const edm::Event& iEvent,const edm::EDGetTokenT<T>& token){
    edm::Handle<T> handle;
    iEvent.getByToken(token,handle);
    return handle;
  }
}

namespace{
  class SCPrinter {
    const reco::SuperCluster& sc_;
  public:
    SCPrinter(const reco::SuperCluster& sc):sc_(sc){}
    std::ostream& operator()(std::ostream& out)const{
      out <<"E "<<sc_.energy()<<" raw E "<<sc_.rawEnergy()<<" eta "<<sc_.eta()<<" phi "<<sc_.phi()<<" seedId "<<sc_.seed()->seed().rawId()<<" nrclus "<<sc_.clustersSize();
      return out;
    }

  };
  std::ostream& operator<<(std::ostream& out,const SCPrinter& obj){return obj(out);}

}

reco::SuperClusterRef matchSCBySeedCrys(const reco::SuperCluster& sc,edm::Handle<reco::SuperClusterCollection> scColl )
{
  for(size_t scNr=0;scNr<scColl->size();scNr++){
    reco::SuperClusterRef scRef(scColl,scNr);
    if(scRef->seed()->seed().rawId() ==sc.seed()->seed().rawId()) return scRef;
  }
  return reco::SuperClusterRef(nullptr,0);
}

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


void EGRefinedSCFixer::produce(edm::Event & iEvent, const edm::EventSetup & iSetup)
{
  auto orgRefinedSCs = getHandle(iEvent,orgRefinedSCToken_);
  auto orgSCs = getHandle(iEvent,orgSCToken_);
  auto fixedSCs = getHandle(iEvent,fixedSCToken_);
  
  std::cout <<"new event "<<std::endl;
  for(auto& sc : *fixedSCs){
    std::cout <<"  fixedSC "<<SCPrinter(sc)<<std::endl;
  }
  for(auto& sc : *orgSCs){
    std::cout <<"  orgSC "<<SCPrinter(sc)<<std::endl;
  }
 

      
  for(size_t scNr=0;scNr<orgRefinedSCs->size();scNr++){
    auto& refinedSC = (*orgRefinedSCs)[scNr];
    if(std::abs(refinedSC.eta())>1.5) continue;
    
    reco::SuperClusterRef orgSC = matchSCBySeedCrys(refinedSC,orgSCs);
    if(orgSC.isNonnull()){
      reco::SuperClusterRef fixedSC = matchSCBySeedCrys(*orgSC,fixedSCs,2,2);
      if(fixedSC.isNonnull()){
	std::cout <<" for org sc "<<SCPrinter(*orgSC)<<" found "<<SCPrinter(*fixedSC)<<std::endl;
	makeFixedRefinedSC(refinedSC,*orgSC,*fixedSC);
      }else{
	std::cout <<" for org sc "<<SCPrinter(*orgSC)<<" did not find fixedSC "<<std::endl;
      }
    }else{
      std::cout <<" for "<<SCPrinter(refinedSC);
      std::cout <<" did not find org "<<std::endl;
    }

  }
  
 

}


std::vector<edm::Ptr<reco::CaloCluster> >
EGRefinedSCFixer::getSubClustersMissing(const reco::SuperCluster& lhs,
					const reco::SuperCluster& rhs)
{
  std::vector<edm::Ptr<reco::CaloCluster> > missingSubClusters;
  for(auto& subClus : lhs.clusters()){
    auto compFunc=[&subClus](const auto& rhs){return subClus->seed()==rhs->seed();};
    if(std::find_if(rhs.clusters().begin(),rhs.clusters().end(),compFunc)==rhs.clusters().end()){ 
      missingSubClusters.push_back(subClus);
    }
  }
  return missingSubClusters;
}

reco::SuperCluster 
EGRefinedSCFixer::makeFixedRefinedSC(const reco::SuperCluster& orgRefinedSC,
				     const reco::SuperCluster& orgSC,
				     const reco::SuperCluster& fixedSC)
{
  auto subClusRefinedAdded = getSubClustersMissing(orgRefinedSC,orgSC);
  auto subClusRefinedRemoved = getSubClustersMissing(orgSC,orgRefinedSC);
  if(!subClusRefinedAdded.empty() || !subClusRefinedRemoved.empty()){
    std::cout <<"for clusters"<<std::endl;
    std::cout <<"org: "<<SCPrinter(orgSC)<<std::endl;
    std::cout <<"refined: "<<SCPrinter(orgRefinedSC)<<std::endl;
    std::cout <<"  sub clusters removed by refined: "<<std::endl;
    for(auto& clus : subClusRefinedRemoved) {
      std::cout <<"    "<<*clus<<std::endl;
    } 
    std::cout <<"  sub clusters added by refined: "<<std::endl;
    for(auto& clus : subClusRefinedAdded) {
      std::cout <<"    "<<*clus<<std::endl;
    }
    std::cout <<std::endl;
  }
 
  return reco::SuperCluster();
    
}
DEFINE_FWK_MODULE(EGRefinedSCFixer);

#endif
