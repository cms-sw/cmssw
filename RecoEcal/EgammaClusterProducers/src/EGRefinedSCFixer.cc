#ifndef RecoEcal_EgammaClusterProducers_EGRefinedSCFixer_h
#define RecoEcal_EgammaClusterProducers_EGRefinedSCFixer_h

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFClusterWidthAlgo.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"

#include "RecoEgamma/EgammaTools/interface/GainSwitchTools.h"

#include <unordered_set>

//work in progress (emergancy fix!)
//issue: we need to re-make the refined superclusters
//we only really need to do this for barrel super with a gain switch
//but usual to remake without gain switch for debugging
//however it is barrel only
//everything else is copied or relinked to produce the same set of outputs
//as particleFlowEGamma

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


  
  static std::unordered_set<unsigned>
  getListOfClusterSeedIdsForNewSC(const reco::SuperCluster& orgRefinedSC,
				  const reco::SuperCluster& orgSC,
				  const reco::SuperCluster& fixedSC,
				  const std::vector<edm::Ptr<reco::CaloCluster> >& clustersAddedToAnySC);
  
  static std::vector<edm::Ptr<reco::PFCluster> >
  getClustersFromSeedIds(const std::unordered_set<unsigned>& seedIds,const edm::Handle<edm::View<reco::PFCluster> >& inClusters);

  static reco::SuperCluster 
  makeFixedRefinedBarrelSC(const reco::SuperCluster& orgRefinedSC,
			   const reco::SuperCluster& orgSC,
			   const reco::SuperCluster& fixedSC,
			   const edm::Handle<edm::View<reco::PFCluster> >& fixedClusters,
			   const std::vector<edm::Ptr<reco::CaloCluster> >& clustersAddedToAnySC);

private:
  typedef edm::View<reco::PFCluster> PFClusterView;
  typedef edm::ValueMap<reco::SuperClusterRef> SCRefMap;
  typedef edm::ValueMap<reco::ConversionRef> ConvRefMap;

  template<typename T>
  void getToken(edm::EDGetTokenT<T>& token,const edm::ParameterSet& pset, const std::string& label, const std::string& instance = "") {
    auto tag(pset.getParameter<edm::InputTag>(label));
    if (!instance.empty())
      tag = edm::InputTag(tag.label(), instance, tag.process());

    token = consumes<T>(tag);
  }

  // outputs from PFEGammaProducer
  edm::EDGetTokenT<reco::SuperClusterCollection> orgRefinedSCToken_;
  edm::EDGetTokenT<reco::CaloClusterCollection> orgBCToken_; // original basic clusters (EB&EE)
  edm::EDGetTokenT<reco::CaloClusterCollection> orgESToken_; // original basic clusters (ES)
  edm::EDGetTokenT<reco::ConversionCollection> orgConvToken_;
  // outputs from PFECALSuperClusterProducer
  edm::EDGetTokenT<reco::SuperClusterCollection> orgSCToken_[2]; // EB & EE+ES
  // outputs from gs-fixed PFECALSuperClusterProducer
  edm::EDGetTokenT<reco::SuperClusterCollection> fixedSCToken_[2]; // EB & EE+ES
  // output from gs-fixed particleFlowClusterECAL
  edm::EDGetTokenT<PFClusterView> fixedPFClustersToken_;

  // output instance name of EB/EE BC collection
  std::string ebeeClustersCollection_;
  // output instance name of ES BC collection
  std::string esClustersCollection_;
};

EGRefinedSCFixer::EGRefinedSCFixer(const edm::ParameterSet& iConfig) :
  ebeeClustersCollection_("EBEEClusters"),
  esClustersCollection_("ESClusters")
{
  getToken(orgRefinedSCToken_, iConfig, "orgRefinedSC");
  getToken(orgBCToken_, iConfig, "orgRefinedSC", "EBEEClusters");
  getToken(orgESToken_, iConfig, "orgRefinedSC", "ESClusters");
  getToken(orgConvToken_, iConfig, "orgRefinedSC");
  getToken(orgSCToken_[0], iConfig, "orgSC", "particleFlowSuperClusterECALBarrel");
  getToken(orgSCToken_[1], iConfig, "orgSC", "particleFlowSuperClusterECALEndcapWithPreshower");
  getToken(fixedSCToken_[0], iConfig, "fixedSC", "particleFlowSuperClusterECALBarrel");
  getToken(fixedSCToken_[1], iConfig, "fixedSC", "particleFlowSuperClusterECALEndcapWithPreshower");
  getToken(fixedPFClustersToken_, iConfig, "fixedPFClusters");

  produces<reco::SuperClusterCollection>();
  produces<reco::CaloClusterCollection>(ebeeClustersCollection_);
  produces<reco::CaloClusterCollection>(esClustersCollection_);  
  produces<reco::ConversionCollection>();

  // products not in PFEGammaProducer; provide mapping from new to old objects
  produces<SCRefMap>();
  produces<SCRefMap>("parentSCsEB");
  produces<SCRefMap>("parentSCsEE");
}
 
namespace {
  template<typename T>
  edm::Handle<T>
  getHandle(const edm::Event& iEvent, const edm::EDGetTokenT<T>& token)
  {
    edm::Handle<T> handle;
    iEvent.getByToken(token, handle);
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
  //  std::ostream& operator<<(std::ostream& out,const SCDeepPrinter& obj){return obj(out);}

} 

void EGRefinedSCFixer::produce(edm::Event & iEvent, const edm::EventSetup & iSetup)
{
  auto orgRefinedSCs = getHandle(iEvent, orgRefinedSCToken_);
  auto orgBCs = getHandle(iEvent, orgBCToken_);
  auto orgESs = getHandle(iEvent, orgESToken_);
  auto orgConvs = getHandle(iEvent, orgConvToken_);
  auto orgEBSCs = getHandle(iEvent,orgSCToken_[0]);
  auto orgEESCs = getHandle(iEvent,orgSCToken_[1]);
  auto fixedEBSCs = getHandle(iEvent,fixedSCToken_[0]);
  auto fixedEESCs = getHandle(iEvent,fixedSCToken_[1]);
  auto fixedPFClusters = getHandle(iEvent,fixedPFClustersToken_);

  auto fixedRefinedSCs = std::make_unique<reco::SuperClusterCollection>();
  // EB clusters are fixed, EE are direct translation from the original
  auto fixedBCs = std::make_unique<reco::CaloClusterCollection>();
  // direct translation of the original ES collection - there is nothing "fixed" about this
  auto fixedESs = std::make_unique<reco::CaloClusterCollection>();
  auto fixedConvs = std::make_unique<reco::ConversionCollection>();

  std::vector<reco::SuperClusterRef> mappedRefinedSCs;
  std::vector<reco::SuperClusterRef> mappedSCsEB(fixedEBSCs->size());
  std::vector<reco::SuperClusterRef> mappedSCsEE(fixedEESCs->size());

  std::vector<edm::Ptr<reco::CaloCluster> > clusterAddedToAnyEBSC;
  for(auto& refinedSC : *orgRefinedSCs){  
    if(refinedSC.seed()->seed().subdetId() == EcalBarrel) { 
      reco::SuperClusterRef parentSC = GainSwitchTools::matchSCBySeedCrys(refinedSC,orgEBSCs);
      if(parentSC.isNonnull()) {
	auto clusAdded = getSubClustersMissing(refinedSC,*parentSC);
	for(auto clusPtr : clusAdded) clusterAddedToAnyEBSC.push_back(clusPtr);
      }else{ //no parent supercluster, all superclusters in this supercluster are addded
	for(auto clusPtr : refinedSC.clusters()) clusterAddedToAnyEBSC.push_back(clusPtr);
      }
    }
  }
    
  for (unsigned iO(0); iO != orgRefinedSCs->size(); ++iO) {
    auto& orgRefinedSC(orgRefinedSCs->at(iO));
    mappedRefinedSCs.emplace_back(orgRefinedSCs, iO);

    // find the PFSuperCluster for this refined SC
    if (orgRefinedSC.seed()->seed().subdetId() == EcalBarrel) {
      auto orgEBSC(GainSwitchTools::matchSCBySeedCrys(orgRefinedSC, orgEBSCs));

      // particleFlowEGamma can create superclusters directly out of PFClusters too
      // -> there is not always a matching EB SC
      if (orgEBSC.isNonnull()) {
        auto fixedEBSC(GainSwitchTools::matchSCBySeedCrys(*orgEBSC, fixedEBSCs, 2, 2));
       
        // here we may genuinely miss a mapping, if the seed position moves too much by re-reconstruction
	// Sam: its very unlikely, if not impossible. To be replaced the gain switched crystal must be within +/-1 crystal a hybrid supercluster seed crystal because of ecalSelectedDigis. You could only get a shift larger than 1 if you had two supercluster seed crystals very close together and even then I'm not sure its possible. 
        if (fixedEBSC.isNonnull()) {
          mappedSCsEB[fixedEBSC.key()] = orgEBSC;

          auto fixedRefinedSC(makeFixedRefinedBarrelSC(orgRefinedSC, *orgEBSC, *fixedEBSC, fixedPFClusters,clusterAddedToAnyEBSC));
          fixedRefinedSCs->push_back(fixedRefinedSC);

	  //	  std::cout<<" org refined "<<SCDeepPrinter(orgRefinedSC)<<std::endl;
	  //std::cout<<" new refined "<<SCDeepPrinter(fixedRefinedSC)<<std::endl;

          continue;
        }
      }
    }
    else {
      auto orgEESC(GainSwitchTools::matchSCBySeedCrys(orgRefinedSC, orgEESCs));
      if (orgEESC.isNonnull()) {
        // there is nothing "fixed" here - two clusters are identical
        auto fixedEESC(GainSwitchTools::matchSCBySeedCrys(*orgEESC, fixedEESCs));
        // fixedEESC has to be nonnull
        mappedSCsEE[fixedEESC.key()] = orgEESC;
      }
    }

    // if EE or no PFSuperCluster match
    fixedRefinedSCs->push_back(orgRefinedSC);
  }


  // Put the PF SC maps in Event
  std::auto_ptr<SCRefMap> pEBSCRefMap(new SCRefMap);
  SCRefMap::Filler ebSCMapFiller(*pEBSCRefMap);
  ebSCMapFiller.insert(fixedEBSCs, mappedSCsEB.begin(), mappedSCsEB.end());
  ebSCMapFiller.fill();
  iEvent.put(pEBSCRefMap, "parentSCsEB");

  std::auto_ptr<SCRefMap> pEESCRefMap(new SCRefMap);
  SCRefMap::Filler eeSCMapFiller(*pEESCRefMap);
  eeSCMapFiller.insert(fixedEESCs, mappedSCsEE.begin(), mappedSCsEE.end());
  eeSCMapFiller.fill();
  iEvent.put(pEESCRefMap, "parentSCsEE");

  // Copy basic clusters
  std::map<reco::CaloClusterPtr, unsigned int> pfClusterMapEBEE; //maps of pfclusters to caloclusters
  std::map<reco::CaloClusterPtr, unsigned int> pfClusterMapES;

  for (auto& sc : *fixedRefinedSCs) {
    // The cluster ref in fixed EB and the other superclusters point to different collections (former to gs-fixed, latter to original)
    // but we are copying the basic clusters by value and remaking yet another collection here -> no need to distinguish
    for (auto&& cItr(sc.clustersBegin()); cItr != sc.clustersEnd(); ++cItr) {
      auto& ptr(*cItr);
      if (pfClusterMapEBEE.count(ptr) == 0) {
        pfClusterMapEBEE[ptr] = fixedBCs->size();
        fixedBCs->emplace_back(*ptr);
      }
      else
        throw cms::Exception("EGRefinedSCFixer::produce")
          << "Found an EB/EE pfcluster matched to more than one supercluster!";
    }

    for (auto&& cItr = sc.preshowerClustersBegin(); cItr!=sc.preshowerClustersEnd(); ++cItr) {
      auto& ptr(*cItr);
      if (pfClusterMapES.count(ptr) == 0) {
        pfClusterMapES[ptr] = fixedESs->size();
        fixedESs->emplace_back(*ptr);
      }
      else
        throw cms::Exception("PFEgammaProducer::produce")
          << "Found an ES pfcluster matched to more than one supercluster!";
    }
  }

  //put calocluster output collections in event and get orphan handles to create ptrs
  auto caloClusHandleEBEE(iEvent.put(std::move(fixedBCs), ebeeClustersCollection_));
  auto caloClusHandleES(iEvent.put(std::move(fixedESs), esClustersCollection_));

  //relink superclusters to output caloclusters
  for (auto& sc : *fixedRefinedSCs) {
    edm::Ptr<reco::CaloCluster> seedptr(caloClusHandleEBEE, pfClusterMapEBEE[sc.seed()]);
    sc.setSeed(seedptr);
    
    reco::CaloClusterPtrVector clusters;
    for (auto&& cItr(sc.clustersBegin()); cItr!=sc.clustersEnd(); ++cItr)
      clusters.push_back(reco::CaloClusterPtr(caloClusHandleEBEE, pfClusterMapEBEE[*cItr]));
    sc.setClusters(clusters);
    
    reco::CaloClusterPtrVector psclusters;
    for (auto&& cItr(sc.preshowerClustersBegin()); cItr!=sc.preshowerClustersEnd(); ++cItr)
      psclusters.push_back(reco::CaloClusterPtr(caloClusHandleES, pfClusterMapES[*cItr]));
    sc.setPreshowerClusters(psclusters);  
  }

  auto scHandle(iEvent.put(std::move(fixedRefinedSCs)));

  // Put the new to old refined SC map
  std::auto_ptr<SCRefMap> pRefinedSCRefMap(new SCRefMap);
  SCRefMap::Filler refinedSCMapFiller(*pRefinedSCRefMap);
  refinedSCMapFiller.insert(scHandle, mappedRefinedSCs.begin(), mappedRefinedSCs.end());
  refinedSCMapFiller.fill();
  iEvent.put(pRefinedSCRefMap);

  // copy and relink conversions
  for (auto& conv : *orgConvs) {
    // copy Ctor is not defined for reco::Conversion but there is a clone() method
    // that uses a copy Ctor. I suppose this object can be trivially copied..
    fixedConvs->emplace_back(conv);
    auto& newConv(fixedConvs->back());
    // conversion keeps a PtrVector, but according to PFEGammaProducer, single-leg conversions
    // always have only one SC ptr.
    reco::CaloClusterPtrVector scPtrVec;
    // we create the new refined SC collection in the same order as the old one
    scPtrVec.push_back(reco::CaloClusterPtr(scHandle, conv.caloCluster()[0].key()));
    newConv.setMatchingSuperCluster(scPtrVec);
  }

  iEvent.put(std::move(fixedConvs));
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

std::unordered_set<unsigned>
EGRefinedSCFixer::getListOfClusterSeedIdsForNewSC(const reco::SuperCluster& orgRefinedSC,
						  const reco::SuperCluster& orgSC,
						  const reco::SuperCluster& fixedSC,
						  const std::vector<edm::Ptr<reco::CaloCluster> >& clustersAddedToAnySC)
  
{
  auto clusAdded = getSubClustersMissing(orgRefinedSC,orgSC);
  auto clusRemoved = getSubClustersMissing(orgSC,orgRefinedSC);

  
  std::unordered_set<unsigned> detIdsOfClustersForNewSC;
  for(auto& clus : fixedSC.clusters()){
    auto compFunc=[&clus](auto& rhs){return rhs->seed().rawId()==clus->seed().rawId();};
    
    //first check if the cluster was removed by the refining process
    bool notRemoved = std::find_if(clusRemoved.begin(),clusRemoved.end(),compFunc)==clusRemoved.end();

    //now check if it was assigned to another supercluster (it wont be picked up by the previous
    //check if the parent supercluster picked it up during reclustering)
    //if its a cluster which is added to any supercluster it wont add it
    //its okay if it was added to this supercluster as we will add it in below
    bool notAddedToASuperCluster = std::find_if(clustersAddedToAnySC.begin(),
						clustersAddedToAnySC.end(),compFunc)==clustersAddedToAnySC.end();
 
    if(notRemoved && notAddedToASuperCluster){
      detIdsOfClustersForNewSC.insert(clus->seed().rawId());
    }
  }
  for(auto clus : clusAdded){
    detIdsOfClustersForNewSC.insert(clus->seed().rawId());
  }
  
  return detIdsOfClustersForNewSC;
}

std::vector<edm::Ptr<reco::PFCluster> >
EGRefinedSCFixer::getClustersFromSeedIds(const std::unordered_set<unsigned>& seedIds,const edm::Handle<edm::View<reco::PFCluster> >& inClusters)
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
					   const edm::Handle<edm::View<reco::PFCluster> >& fixedClusters,
					   const std::vector<edm::Ptr<reco::CaloCluster> >& clustersAddedToAnySC)
{
  auto listOfSeedIds = getListOfClusterSeedIdsForNewSC(orgRefinedSC,orgSC,fixedSC,clustersAddedToAnySC);

  // make sure the seed cluster is in the listOfSeedIds
  unsigned seedSeedId(fixedSC.seed()->seed().rawId());
  listOfSeedIds.insert(seedSeedId);

  std::vector<edm::Ptr<reco::PFCluster> > clusters = getClustersFromSeedIds(listOfSeedIds,fixedClusters);

  // fixedSC.seed() is from a different BasicCluster collection
  edm::Ptr<reco::PFCluster> seedCluster;
  for (auto& ptr : clusters) {
    if (ptr->seed().rawId() == seedSeedId) {
      seedCluster = ptr;
      break;
    }
  }

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
  newSC.setSeed(seedCluster); //the seed is the same as the non-refined SC
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
