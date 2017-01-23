#ifndef RecoEcal_EgammaClusterProducers_EGRefinedSCFixer_h
#define RecoEcal_EgammaClusterProducers_EGRefinedSCFixer_h

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

#include "RecoEgamma/EgammaTools/interface/GainSwitchTools.h"

#include <unordered_set>

//work in progress (emergancy fix!)
//issue: we need to re-make the refined superclusters
//we only really need to do this for barrel super with a gain switch
//but usual to remake without gain switch for debugging
//however it is barrel only

//how it works:
//  a refined supercluster will have subclusters added or removed w.r.t to its parent SC
//  we cant remake them in AOD (yet)
//  so we take the existing refined superclusters and see what sub clusters are removed/added 
//  and then when we take our new fixed superclusters, we add or remove those clusters
//  we id clusters via seed crystal, this shouldnt change for any cluster without a gain switch, 
//  its still a local maximum
//  
//  for matching the new fixed superclusters vs old superclusters, the seed crystal may gain
//  as the gain switched crystal will now have a larger energy
//  but it should be within the 5x5 (really the 3x3 of the orginal SC)
//  so we do a dIR = dIEta^2 + dIPhi^2 match of <=8 (ie be in the 5x5)
//  but take the smallest dIR

//  issues: sub cluster ordering may not be correct (its sorted by decreasing energy)
//  issues: when it assigns a sub cluster to a refined SC, it doesnt remove it from others

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
  
  static std::vector<edm::Ptr<reco::PFCluster> >
  getClustersFromSeedIds(const std::unordered_set<int>& seedIds,const edm::Handle<edm::View<reco::PFCluster> >& inClusters);

  static reco::SuperCluster 
  makeFixedRefinedBarrelSC(const reco::SuperCluster& orgRefinedSC,
			   const reco::SuperCluster& orgSC,
			   const reco::SuperCluster& fixedSC,
			   const edm::Handle<edm::View<reco::PFCluster> >& fixedClusters);

private:
  edm::EDGetTokenT<reco::SuperClusterCollection> orgRefinedSCToken_;
  edm::EDGetTokenT<reco::SuperClusterCollection> orgSCToken_;
  edm::EDGetTokenT<reco::SuperClusterCollection> fixedSCToken_;
  edm::EDGetTokenT<edm::View<reco::PFCluster> > fixedPFClustersToken_;
  
  

};

EGRefinedSCFixer::EGRefinedSCFixer(const edm::ParameterSet& iConfig )
{
  orgRefinedSCToken_ = consumes<reco::SuperClusterCollection>(iConfig.getParameter<edm::InputTag>("orgRefinedSC"));
  orgSCToken_ = consumes<reco::SuperClusterCollection>(iConfig.getParameter<edm::InputTag>("orgSC"));
  fixedSCToken_ = consumes<reco::SuperClusterCollection>(iConfig.getParameter<edm::InputTag>("fixedSC"));
  fixedPFClustersToken_ = consumes<edm::View<reco::PFCluster> >(iConfig.getParameter<edm::InputTag>("fixedPFClusters"));

  produces<reco::SuperClusterCollection>(); 
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
  // std::ostream& operator<<(std::ostream& out,const SCPrinter& obj){return obj(out);}

}

namespace{
  class SCDeepPrinter {
    const reco::SuperCluster& sc_;
  public:
    SCDeepPrinter(const reco::SuperCluster& sc):sc_(sc){}
    std::ostream& operator()(std::ostream& out)const{
      out <<"E "<<sc_.energy()<<" raw E "<<sc_.rawEnergy()<<" eta "<<sc_.eta()<<" phi "<<sc_.phi()<<" seedId "<<sc_.seed()->seed().rawId()<<" nrclus "<<sc_.clustersSize() <<std::endl;
      for(auto& clus : sc_.clusters()){
	out <<" clus E "<<clus->energy()<<" Ecorr "<<clus->correctedEnergy()<<" seed "<<clus->seed().rawId()<<" nrHits "<<clus->hitsAndFractions().size()<<std::endl;
      }
      return out;
    }

  };
  std::ostream& operator<<(std::ostream& out,const SCDeepPrinter& obj){return obj(out);}

}

void EGRefinedSCFixer::produce(edm::Event & iEvent, const edm::EventSetup & iSetup)
{
  auto orgRefinedSCs = getHandle(iEvent,orgRefinedSCToken_);
  auto orgSCs = getHandle(iEvent,orgSCToken_);
  auto fixedSCs = getHandle(iEvent,fixedSCToken_);
  auto fixedPFClusters = getHandle(iEvent,fixedPFClustersToken_);
  
  
  // std::cout <<"new event "<<std::endl;
  // for(auto& sc : *fixedSCs){
  //   std::cout <<"  fixedSC "<<SCPrinter(sc)<<std::endl;
  // }
  // for(auto& sc : *orgSCs){
  //   std::cout <<"  orgSC "<<SCPrinter(sc)<<std::endl;
  // }
   
  auto fixedRefinedSCs = std::make_unique<reco::SuperClusterCollection>();
  
  
  for(const auto& fixedSC : *fixedSCs){
    
    reco::SuperClusterRef orgSC = GainSwitchTools::matchSCBySeedCrys(fixedSC,orgSCs,2,2);
    reco::SuperClusterRef orgRefinedSC = orgSC.isNonnull() ? GainSwitchTools::matchSCBySeedCrys(*orgSC,orgRefinedSCs) : orgSC;
    
    //so we have a matched orginal and refined SC, we can remake the refined SC out of the new clusters
    if(orgSC.isNonnull() && orgRefinedSC.isNonnull()){
      reco::SuperCluster fixedRefinedSC = makeFixedRefinedBarrelSC(*orgRefinedSC,*orgSC,fixedSC,fixedPFClusters);
      if(orgRefinedSC->clustersSize()!=orgSC->clustersSize() && false){
	std::cout<<" org refined "<<SCDeepPrinter(*orgRefinedSC)<<std::endl;
	std::cout<<" fixed refined "<<SCDeepPrinter(fixedRefinedSC)<<std::endl;
      }
      fixedRefinedSCs->push_back(fixedRefinedSC);
    }else if(orgSC.isNull()){
      //didnt find the orginal supercluster, just pass this through as a refined sc
      fixedRefinedSCs->push_back(fixedSC);      
    }//else{
      //      std::cout <<"for "<<SCPrinter(fixedSC)<<" did not find orgSC "<<orgSC.isNonnull()<<" refined "<<orgRefinedSC.isNonnull()<<std::endl;
    //}
  }
  iEvent.put(std::move(fixedRefinedSCs));

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

std::vector<edm::Ptr<reco::PFCluster> >
EGRefinedSCFixer::getClustersFromSeedIds(const std::unordered_set<int>& seedIds,const edm::Handle<edm::View<reco::PFCluster> >& inClusters)
{
  std::vector<edm::Ptr<reco::PFCluster> > outClusters;
  for(size_t clusNr=0;clusNr<inClusters->size();clusNr++){
    edm::Ptr<reco::PFCluster> clusPtr(inClusters,clusNr);
    if(seedIds.count(clusPtr->seed().rawId())>0){
      outClusters.push_back(clusPtr);
    }
  }
  std::sort(outClusters.begin(),outClusters.end(),[](auto& lhs,auto& rhs){return lhs->energy()>rhs->energy();});
  return outClusters;
}

//EB only which simplies things a lot
//stolen from PFEGammaAlgo which stole it from PFECALSuperClusterAlgo
//probably some sort of refactoring would be handy
reco::SuperCluster 
EGRefinedSCFixer::makeFixedRefinedBarrelSC(const reco::SuperCluster& orgRefinedSC,
					   const reco::SuperCluster& orgSC,
					   const reco::SuperCluster& fixedSC,
					   const edm::Handle<edm::View<reco::PFCluster> >& fixedClusters)
{
  
  auto listOfSeedIds = getListOfClusterSeedIdsForNewSC(orgRefinedSC,orgSC,fixedSC);
  std::vector<edm::Ptr<reco::PFCluster> > clusters = getClustersFromSeedIds(listOfSeedIds,fixedClusters);
  std::vector<const reco::PFCluster*> clustersBarePtrs;

  double posX(0),posY(0),posZ(0);
  double scNrgy(0),scCorrNrgy(0);
  for(auto & clus : clusters){
    clustersBarePtrs.push_back(&*clus);
    
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
  PFClusterWidthAlgo pfwidth(clustersBarePtrs);
  newSC.setEtaWidth(pfwidth.pflowEtaWidth());
  newSC.setPhiWidth(pfwidth.pflowPhiWidth());
  
  // cache the value of the raw energy  
  newSC.rawEnergy();
  
  return newSC;
    
}

DEFINE_FWK_MODULE(EGRefinedSCFixer);

#endif
