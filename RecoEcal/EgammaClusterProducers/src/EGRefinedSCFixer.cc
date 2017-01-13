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
#include "RecoParticleFlow/PFClusterTools/interface/PFClusterWidthAlgo.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include <unordered_set>

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


  
  static std::unordered_set<int> 
  getListOfClusterSeedIdsForNewSC(const reco::SuperCluster& orgRefinedSC,
				  const reco::SuperCluster& orgSC,
				  const reco::SuperCluster& fixedSC);
  
  static std::vector<edm::Ptr<reco::CaloCluster> >
  getClustersFromSeedIds(const std::unordered_set<int>& seedIds,const edm::Handle<edm::View<reco::CaloCluster> >& inClusters);

  static reco::SuperCluster 
  makeFixedRefinedBarrelSC(const reco::SuperCluster& orgRefinedSC,
			   const reco::SuperCluster& orgSC,
			   const reco::SuperCluster& fixedSC,
			   const edm::Handle<edm::View<reco::CaloCluster> >& fixedClusters);

private:
  edm::EDGetTokenT<reco::SuperClusterCollection> orgRefinedSCToken_;
  edm::EDGetTokenT<reco::SuperClusterCollection> orgSCToken_;
  edm::EDGetTokenT<reco::SuperClusterCollection> fixedSCToken_;
  edm::EDGetTokenT<edm::View<reco::CaloCluster> > fixedPFClustersToken_;
  
  

};

EGRefinedSCFixer::EGRefinedSCFixer(const edm::ParameterSet& iConfig )
{
  orgRefinedSCToken_ = consumes<reco::SuperClusterCollection>(iConfig.getParameter<edm::InputTag>("orgRefinedSC"));
  orgSCToken_ = consumes<reco::SuperClusterCollection>(iConfig.getParameter<edm::InputTag>("orgSC"));
  fixedSCToken_ = consumes<reco::SuperClusterCollection>(iConfig.getParameter<edm::InputTag>("fixedSC"));
  fixedPFClustersToken_ = consumes<edm::View<reco::CaloCluster> >(iConfig.getParameter<edm::InputTag>("fixedPFClusters"));
  
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
  auto fixedPFClusters = getHandle(iEvent,fixedPFClustersToken_);
  
  
  std::cout <<"new event "<<std::endl;
  for(auto& sc : *fixedSCs){
    std::cout <<"  fixedSC "<<SCPrinter(sc)<<std::endl;
  }
  for(auto& sc : *orgSCs){
    std::cout <<"  orgSC "<<SCPrinter(sc)<<std::endl;
  }
   
  for(const auto& fixedSC : *fixedSCs){
    
    reco::SuperClusterRef orgSC = matchSCBySeedCrys(fixedSC,orgSCs,2,2);
    reco::SuperClusterRef orgRefinedSC = orgSC.isNonnull() ? matchSCBySeedCrys(*orgSC,orgRefinedSCs) : orgSC;
    
    //so we have a matched orginal and refined SC, we can remake the refined SC out of the new clusters
    if(orgSC.isNonnull() && orgRefinedSC.isNonnull()){
      reco::SuperCluster fixedRefinedSC = makeFixedRefinedBarrelSC(*orgRefinedSC,*orgSC,fixedSC,fixedPFClusters);
      std::cout<<" org refined "<<*orgRefinedSC<<std::endl;
      std::cout<<" fixed refined "<<fixedRefinedSC<<std::endl;
      
    }else{
      std::cout <<"interesting, orgSC "<<orgSC.isNonnull()<<" refined "<<orgRefinedSC.isNonnull()<<std::endl;
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

std::unordered_set<int> 
EGRefinedSCFixer::getListOfClusterSeedIdsForNewSC(const reco::SuperCluster& orgRefinedSC,
						  const reco::SuperCluster& orgSC,
						  const reco::SuperCluster& fixedSC)
  
{
  auto clusAdded = getSubClustersMissing(orgRefinedSC,orgSC);
  auto clusRemoved = getSubClustersMissing(orgSC,orgRefinedSC);
  
  std::unordered_set<int> detIdsOfClustersForNewSC;
  for(auto& clus : fixedSC.clusters()){
    auto compFunc=[&clus](auto& rhs){return rhs->seed().rawId()==clus->seed().rawId();};
    if(std::find_if(clusRemoved.begin(),clusRemoved.end(),compFunc)==clusRemoved.end()){
      detIdsOfClustersForNewSC.insert(clus->seed().rawId());
    }
  }
  for(auto clus : clusAdded) detIdsOfClustersForNewSC.insert(clus->seed().rawId());
  
  return detIdsOfClustersForNewSC;
}

std::vector<edm::Ptr<reco::CaloCluster> >
EGRefinedSCFixer::getClustersFromSeedIds(const std::unordered_set<int>& seedIds,const edm::Handle<edm::View<reco::CaloCluster> >& inClusters)
{
  std::vector<edm::Ptr<reco::CaloCluster> > outClusters;
  for(size_t clusNr=0;clusNr<inClusters->size();clusNr++){
    edm::Ptr<reco::CaloCluster> clusPtr(inClusters,clusNr);
    if(seedIds.count(clusPtr->seed().rawId())>0){
      outClusters.push_back(clusPtr);
    }
  }
  return outClusters;
}

//EB only which simplies things a lot
//stolen from PFEGammaAlgo which stole it from PFECALSuperClusterAlgo
//probably some sort of refactoring would be handy
reco::SuperCluster 
EGRefinedSCFixer::makeFixedRefinedBarrelSC(const reco::SuperCluster& orgRefinedSC,
					   const reco::SuperCluster& orgSC,
					   const reco::SuperCluster& fixedSC,
					   const edm::Handle<edm::View<reco::CaloCluster> >& fixedClusters)
{
  
  auto listOfSeedIds = getListOfClusterSeedIdsForNewSC(orgRefinedSC,orgSC,fixedSC);
  std::vector<edm::Ptr<reco::CaloCluster> > clusters = getClustersFromSeedIds(listOfSeedIds,fixedClusters);
  //std::vector<const reco::PFCluster*> clustersBarePtrs;

  double posX(0),posY(0),posZ(0);
  double scNrgy(0),scCorrNrgy(0);
  for(auto & clus : clusters){
    //    clustersBarePtrs.push_back(&*clus);
    
    const double clusNrgy = clus->energy();
    double clusCorrNrgy = clus->correctedEnergy();
    const math::XYZPoint& clusPos = clus->position();
    posX += clusNrgy * clusPos.X();
    posY += clusNrgy * clusPos.Y();
    posZ += clusNrgy * clusPos.Z();
        
    scNrgy  += clusNrgy;
    scCorrNrgy += clusCorrNrgy;    

  }
  posX /= scNrgy;
  posY /= scNrgy;
  posZ /= scNrgy;

  //intentionally passing in scCorrNrgy its not supposed to be scNrgy
  reco::SuperCluster newSC(scCorrNrgy,math::XYZPoint(posX,posY,posZ));

  
  newSC.setCorrectedEnergy(scCorrNrgy);
  newSC.setSeed(fixedSC.seed()); //the seed is the same as the non-refined SC
  newSC.setPreshowerEnergyPlane1(0.);
  newSC.setPreshowerEnergyPlane2(0.);
  newSC.setPreshowerEnergy(0.); 
  for(const auto& clus : clusters ) {
    newSC.addCluster(clus);
    for(auto& hitAndFrac: clus->hitsAndFractions() ) {
      newSC.addHitAndFraction(hitAndFrac.first,hitAndFrac.second);
    }
  }
  
  // calculate linearly weighted cluster widths
  //PFClusterWidthAlgo pfwidth(clustersBarePtrs);
  //newSC.setEtaWidth(pfwidth.pflowEtaWidth());
  //newSC.setPhiWidth(pfwidth.pflowPhiWidth());
  
  // cache the value of the raw energy  
  newSC.rawEnergy();
  
  return newSC;
    
}
DEFINE_FWK_MODULE(EGRefinedSCFixer);

#endif
