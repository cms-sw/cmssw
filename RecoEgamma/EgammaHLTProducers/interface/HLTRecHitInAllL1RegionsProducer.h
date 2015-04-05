#ifndef RecoEgamma_EgammayHLTProducers_HLTRecHitInAllL1RegionsProducer_h_
#define RecoEgamma_EgammayHLTProducers_HLTRecHitInAllL1RegionsProducer_h_

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

// Reco candidates (might not need)
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"

// Geometry and topology
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

#include "RecoEcal/EgammaCoreTools/interface/EcalEtaPhiRegion.h"
// Level 1 Trigger
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h" 
#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"

#include "CondFormats/L1TObjects/interface/L1CaloGeometry.h"
#include "CondFormats/DataRecord/interface/L1CaloGeometryRecord.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"


#include "HLTrigger/HLTcore/interface/defaultModuleLabel.h"

//this is a re-write of HLTRechitInRegionsProducer to be able to handle arbitary L1 collections as inputs
//in the process, some of the cruft was cleaned up but it mantains almost all the old behaviour
//think the only difference now is that it wont throw if its not ECALBarrel, ECALEndcap or ECAL PS rec-hit type
class L1RegionDataBase {
public:
  virtual ~L1RegionDataBase(){}
  virtual void getEtaPhiRegions(const edm::Event&,std::vector<EcalEtaPhiRegion>&,const L1CaloGeometry&)const=0;
};  

template<typename T1> class L1RegionData : public L1RegionDataBase {
private:
  double minEt_;
  double maxEt_;
  double regionEtaMargin_;
  double regionPhiMargin_;
  edm::EDGetTokenT<T1> token_;
public:
  L1RegionData(const edm::ParameterSet& para,edm::ConsumesCollector & consumesColl):
    minEt_(para.getParameter<double>("minEt")),
    maxEt_(para.getParameter<double>("maxEt")),
    regionEtaMargin_(para.getParameter<double>("regionEtaMargin")),
    regionPhiMargin_(para.getParameter<double>("regionPhiMargin")),
    token_(consumesColl.consumes<T1>(para.getParameter<edm::InputTag>("inputColl"))){}
  
  void getEtaPhiRegions(const edm::Event&,std::vector<EcalEtaPhiRegion>&,const L1CaloGeometry&)const override;
};



template<typename RecHitType>
class HLTRecHitInAllL1RegionsProducer : public edm::stream::EDProducer<> {
  
  using  RecHitCollectionType =edm::SortedCollection<RecHitType>;
  
 public:

  HLTRecHitInAllL1RegionsProducer(const edm::ParameterSet& ps);
  ~HLTRecHitInAllL1RegionsProducer(){}

  void produce(edm::Event&, const edm::EventSetup&) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

 private:
  L1RegionDataBase* createL1RegionData(const std::string&,const edm::ParameterSet&,edm::ConsumesCollector &&); //calling function owns this
  
  std::vector<std::unique_ptr<L1RegionDataBase>> l1RegionData_;
 
  std::vector<edm::InputTag> recHitLabels_;
  std::vector<std::string> productLabels_;

  std::vector<edm::EDGetTokenT<RecHitCollectionType>> recHitTokens_;

  
};


template<typename RecHitType> 
HLTRecHitInAllL1RegionsProducer<RecHitType>::HLTRecHitInAllL1RegionsProducer(const edm::ParameterSet& para)
{
  const std::vector<edm::ParameterSet> l1InputRegions = para.getParameter<std::vector<edm::ParameterSet>>("l1InputRegions");
  for(auto& pset : l1InputRegions){
    const std::string type=pset.getParameter<std::string>("type");
    l1RegionData_.emplace_back(createL1RegionData(type,pset,consumesCollector())); //meh I was going to use a factory but it was going to be overly complex for my needs
  }
  recHitLabels_ =para.getParameter<std::vector<edm::InputTag>>("recHitLabels");
  productLabels_=para.getParameter<std::vector<std::string>>("productLabels");

  for (unsigned int collNr=0; collNr<recHitLabels_.size(); collNr++) { 
    recHitTokens_.push_back(consumes<RecHitCollectionType>(recHitLabels_[collNr]));
    produces<RecHitCollectionType> (productLabels_[collNr]);
  }
}
template<typename RecHitType> 
void HLTRecHitInAllL1RegionsProducer<RecHitType>::fillDescriptions(edm::ConfigurationDescriptions& descriptions)
{
  edm::ParameterSetDescription desc;
  std::vector<std::string> productTags;
  productTags.push_back("EcalRegionalRecHitsEB");
  productTags.push_back("EcalRegionalRecHitsEE");
  desc.add<std::vector<std::string>>("productLabels", productTags);
  std::vector<edm::InputTag> recHitLabels;
  recHitLabels.push_back(edm::InputTag("hltEcalRegionalEgammaRecHit:EcalRecHitsEB"));
  recHitLabels.push_back(edm::InputTag("hltEcalRegionalEgammaRecHit:EcalRecHitsEE"));
  recHitLabels.push_back(edm::InputTag("hltESRegionalEgammaRecHit:EcalRecHitsES"));
  desc.add<std::vector<edm::InputTag>>("recHitLabels", recHitLabels);
  std::vector<edm::ParameterSet> l1InputRegions;
  edm::ParameterSet emIsoPSet;
  emIsoPSet.addParameter<std::string>("type","L1EmParticle");
  emIsoPSet.addParameter<double>("minEt",5);
  emIsoPSet.addParameter<double>("maxEt",999);
  emIsoPSet.addParameter<double>("regionEtaMargin",0.14);
  emIsoPSet.addParameter<double>("regionPhiMargin",0.4);
  emIsoPSet.addParameter<edm::InputTag>("inputColl",edm::InputTag("hltL1extraParticles:NonIsolated"));
  l1InputRegions.push_back(emIsoPSet);
  edm::ParameterSet emNonIsoPSet;
  emNonIsoPSet.addParameter<std::string>("type","L1EmParticle");
  emNonIsoPSet.addParameter<double>("minEt",5);
  emNonIsoPSet.addParameter<double>("maxEt",999);
  emNonIsoPSet.addParameter<double>("regionEtaMargin",0.14);
  emNonIsoPSet.addParameter<double>("regionPhiMargin",0.4);
  emNonIsoPSet.addParameter<edm::InputTag>("inputColl",edm::InputTag("hltL1extraParticles:Isolated"));
  l1InputRegions.push_back(emNonIsoPSet);
  
  edm::ParameterSetDescription l1InputRegionDesc;
  l1InputRegionDesc.add<std::string>("type");
  l1InputRegionDesc.add<double>("minEt");
  l1InputRegionDesc.add<double>("maxEt");
  l1InputRegionDesc.add<double>("regionEtaMargin");
  l1InputRegionDesc.add<double>("regionPhiMargin");
  l1InputRegionDesc.add<edm::InputTag>("inputColl");
  desc.addVPSet("l1InputRegions",l1InputRegionDesc,l1InputRegions);
  
  descriptions.add(defaultModuleLabel<HLTRecHitInAllL1RegionsProducer<RecHitType>>(), desc); 
}


template<typename RecHitType>
void HLTRecHitInAllL1RegionsProducer<RecHitType>::produce(edm::Event& event, const edm::EventSetup& setup) {

  // get the collection geometry:
  edm::ESHandle<CaloGeometry> caloGeomHandle;
  setup.get<CaloGeometryRecord>().get(caloGeomHandle);
    
   // Get the CaloGeometry
  edm::ESHandle<L1CaloGeometry> l1CaloGeom ;
  setup.get<L1CaloGeometryRecord>().get(l1CaloGeom) ;
  
  std::vector<EcalEtaPhiRegion> regions;
  std::for_each(l1RegionData_.begin(),l1RegionData_.end(),
		[&event,&regions,l1CaloGeom](const std::unique_ptr<L1RegionDataBase>& input)
		{input->getEtaPhiRegions(event,regions,*l1CaloGeom);}
		);
    
  for(size_t recHitCollNr=0;recHitCollNr<recHitTokens_.size();recHitCollNr++){
    edm::Handle<RecHitCollectionType> recHits;
    event.getByToken(recHitTokens_[recHitCollNr],recHits);
    
    if (!(recHits.isValid())) {
      edm::LogError("ProductNotFound")<< "could not get a handle on the "<<typeid(RecHitCollectionType).name() <<" named "<< recHitLabels_[recHitCollNr].encode() << std::endl;
      continue;
    }

    std::auto_ptr<RecHitCollectionType> filteredRecHits(new RecHitCollectionType);
      
    if(!recHits->empty()){
      const CaloSubdetectorGeometry* subDetGeom=caloGeomHandle->getSubdetectorGeometry(recHits->front().id());
      if(!regions.empty()){
      
	for(const RecHitType& recHit : *recHits){
	  const CaloCellGeometry* recHitGeom = subDetGeom->getGeometry(recHit.id());
	  GlobalPoint position = recHitGeom->getPosition();
	  for(const auto& region : regions){
	    if(region.inRegion(position)) {
	      filteredRecHits->push_back(recHit);
		break;
	    }
	  }
	}
      }//end check of empty regions
    }//end check of empty rec-hits
    //   std::cout <<"putting fileter coll in "<<filteredRecHits->size()<<std::endl;
    event.put(filteredRecHits,productLabels_[recHitCollNr]);
  }//end loop over all rec hit collections

}





template<typename RecHitType> 
L1RegionDataBase* HLTRecHitInAllL1RegionsProducer<RecHitType>::createL1RegionData(const std::string& type,const edm::ParameterSet& para,edm::ConsumesCollector && consumesColl)
{
  if(type=="L1EmParticle"){
    return new L1RegionData<l1extra::L1EmParticleCollection>(para,consumesColl);
  }else if(type=="L1JetParticle"){
    return new L1RegionData<l1extra::L1JetParticleCollection>(para,consumesColl);
  }else if(type=="L1MuonParticle"){
    return new L1RegionData<l1extra::L1MuonParticleCollection>(para,consumesColl);
  }else{
    //this is a major issue and could lead to rather subtle efficiency losses, so if its incorrectly configured, we're aborting the job!
    throw cms::Exception("InvalidConfig") << " type "<<type<<" is not recognised, this means the rec-hit you think you are keeping may not be and you should fix this error as it can lead to hard to find efficiency loses"<<std::endl;
  }

}



template<typename L1CollType>
void L1RegionData<L1CollType>::getEtaPhiRegions(const edm::Event& event,std::vector<EcalEtaPhiRegion>&regions,const L1CaloGeometry&)const
{
  edm::Handle<L1CollType> l1Cands;
  event.getByToken(token_,l1Cands);
  
  for(const auto& l1Cand : *l1Cands){
    if(l1Cand.et() >= minEt_ && l1Cand.et() < maxEt_){
      
      double etaLow = l1Cand.eta() - regionEtaMargin_;
      double etaHigh = l1Cand.eta() + regionEtaMargin_;
      double phiLow = l1Cand.phi() - regionPhiMargin_;
      double phiHigh = l1Cand.phi() + regionPhiMargin_;
      
      regions.push_back(EcalEtaPhiRegion(etaLow,etaHigh,phiLow,phiHigh));
    }
  }
}

template<>
void L1RegionData<l1extra::L1JetParticleCollection>::getEtaPhiRegions(const edm::Event& event,std::vector<EcalEtaPhiRegion>&regions,const L1CaloGeometry& l1CaloGeom)const
{
  edm::Handle<l1extra::L1JetParticleCollection> l1Cands;
  event.getByToken(token_,l1Cands);
  
  for(const auto& l1Cand : *l1Cands){
    if(l1Cand.et() >= minEt_ && l1Cand.et() < maxEt_){
     
      // Access the GCT hardware object corresponding to the L1Extra EM object.
      int etaIndex = l1Cand.gctJetCand()->etaIndex();
      int phiIndex = l1Cand.gctJetCand()->phiIndex();
      
      // Use the L1CaloGeometry to find the eta, phi bin boundaries.
      double etaLow  = l1CaloGeom.etaBinLowEdge(etaIndex);
      double etaHigh = l1CaloGeom.etaBinHighEdge(etaIndex);
      double phiLow  = l1CaloGeom.emJetPhiBinLowEdge( phiIndex ) ;
      double phiHigh = l1CaloGeom.emJetPhiBinHighEdge( phiIndex ) ;
      
      etaLow -= regionEtaMargin_;
      etaHigh += regionEtaMargin_;
      phiLow -= regionPhiMargin_;
      phiHigh += regionPhiMargin_;

      
      regions.push_back(EcalEtaPhiRegion(etaLow,etaHigh,phiLow,phiHigh));
    }
  }
}

template<>
void L1RegionData<l1extra::L1EmParticleCollection>::getEtaPhiRegions(const edm::Event& event,std::vector<EcalEtaPhiRegion>&regions,const L1CaloGeometry& l1CaloGeom)const
{
  edm::Handle<l1extra::L1EmParticleCollection> l1Cands;
  event.getByToken(token_,l1Cands);
  
  for(const auto& l1Cand : *l1Cands){
    if(l1Cand.et() >= minEt_ && l1Cand.et() < maxEt_){
      
       // Access the GCT hardware object corresponding to the L1Extra EM object.
      int etaIndex = l1Cand.gctEmCand()->etaIndex();
      int phiIndex = l1Cand.gctEmCand()->phiIndex();
      
      // Use the L1CaloGeometry to find the eta, phi bin boundaries.
      double etaLow  = l1CaloGeom.etaBinLowEdge(etaIndex);
      double etaHigh = l1CaloGeom.etaBinHighEdge(etaIndex);
      double phiLow  = l1CaloGeom.emJetPhiBinLowEdge( phiIndex ) ;
      double phiHigh = l1CaloGeom.emJetPhiBinHighEdge( phiIndex ) ;
      
      etaLow -= regionEtaMargin_;
      etaHigh += regionEtaMargin_;
      phiLow -= regionPhiMargin_;
      phiHigh += regionPhiMargin_;
      
      regions.push_back(EcalEtaPhiRegion(etaLow,etaHigh,phiLow,phiHigh));
    }
  }
}




typedef HLTRecHitInAllL1RegionsProducer<EcalRecHit> HLTEcalRecHitInAllL1RegionsProducer;
DEFINE_FWK_MODULE(HLTEcalRecHitInAllL1RegionsProducer);
typedef HLTRecHitInAllL1RegionsProducer<EcalUncalibratedRecHit> HLTEcalUncalibratedRecHitInAllL1RegionsProducer;
DEFINE_FWK_MODULE(HLTEcalUncalibratedRecHitInAllL1RegionsProducer);


#endif


